"""Rebuild go-g2p's embedded data and ONNX models from their pinned sources.

Downloads are pinned by URL and SHA-256. Outputs are written in the repository
layout (data/, models/, fallback/neural/models/) under --output. With --check,
each output is compared byte-for-byte with the committed file.
"""

import argparse
import hashlib
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import spacy
import torch
from transformers import BartForConditionalGeneration

REPO = Path(__file__).resolve().parents[2]
MISAKI = "https://raw.githubusercontent.com/hexgrad/misaki/fba1236595f2d2bf21d414ba6e57d25256afada3/misaki/data/"
NEURAL = {
    "us": "https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us/resolve/a5631b285d18d59483c32c0c3379cb9fac924f4b/",
    "gb": "https://huggingface.co/PeterReid/graphemes_to_phonemes_en_gb/resolve/d8357d5067fa26a5c34134d6bbcf4bbf000c0ac8/",
}
SOURCES = {
    "misaki/us_gold.json": (MISAKI + "us_gold.json", "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f"),
    "misaki/us_silver.json": (MISAKI + "us_silver.json", "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475"),
    "misaki/gb_gold.json": (MISAKI + "gb_gold.json", "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71"),
    "misaki/gb_silver.json": (MISAKI + "gb_silver.json", "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce"),
    "us/config.json": (NEURAL["us"] + "config.json", "8deb3537fb29c63cd9f20d75515ae06e4c92f1b6db0703a2d45bca95b33a53a4"),
    "us/generation_config.json": (NEURAL["us"] + "generation_config.json", "79a11240b01c6d34cab0b122b63a2b2dd5938c4d37d1530e14144d27e8ba6272"),
    "us/model.safetensors": (NEURAL["us"] + "model.safetensors", "dc4a02e62d4fcb4bb4097ecf00db89b8e1a12a549a52ab6adfbba220b80a55c5"),
    "gb/config.json": (NEURAL["gb"] + "config.json", "e4f248e6af0cfb6cb54aea6ff0168d16b6f3cdabed438c487cde4ab0815d1bac"),
    "gb/generation_config.json": (NEURAL["gb"] + "generation_config.json", "79a11240b01c6d34cab0b122b63a2b2dd5938c4d37d1530e14144d27e8ba6272"),
    "gb/model.safetensors": (NEURAL["gb"] + "model.safetensors", "4994f474bb6f4584076a4e98189caaec11aa3f773a0d82d5bc263a03dd07e703"),
}
# Output paths relative to the repository root.
OUTPUTS = [
    "data/us_gold.json",
    "data/us_silver.json",
    "data/gb_gold.json",
    "data/gb_silver.json",
    "data/tokenizer.json",
    "data/unicode.json",
    "models/pos.json",
    "models/pos.onnx",
    "fallback/neural/models/us.json",
    "fallback/neural/models/us-encoder.onnx",
    "fallback/neural/models/us-decoder.onnx",
    "fallback/neural/models/gb.json",
    "fallback/neural/models/gb-encoder.onnx",
    "fallback/neural/models/gb-decoder.onnx",
]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(cache):
    """Download each pinned source once and verify its hash."""
    for name, (url, digest) in SOURCES.items():
        target = cache / name
        if target.is_file() and sha256(target) == digest:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                while block := response.read(1 << 20):
                    out.write(block)
            if sha256(partial) != digest:
                raise RuntimeError(f"hash mismatch for {url}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def export(model, inputs, path, names, outputs, axes):
    with torch.no_grad():
        torch.onnx.export(
            model.eval(),
            inputs,
            str(path),
            input_names=names,
            output_names=outputs,
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(str(path), full_check=True)


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, ids):
        return self.encoder(input_ids=ids, return_dict=False)[0]


class Decoder(torch.nn.Module):
    """One greedy step: logits for the last position only."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.head = model.lm_head
        self.register_buffer("bias", model.final_logits_bias)

    def forward(self, ids, hidden):
        decoded = self.decoder(
            input_ids=ids,
            encoder_hidden_states=hidden,
            use_cache=False,
            return_dict=False,
        )[0]
        return self.head(decoded[:, -1, :]) + self.bias


class Maxout(torch.nn.Module):
    def __init__(self, maxout, norm):
        super().__init__()
        for name, data in [
            ("w", maxout.get_param("W")),
            ("b", maxout.get_param("b")),
            ("gain", norm.get_param("G")),
            ("bias", norm.get_param("b")),
        ]:
            self.register_buffer(name, torch.from_numpy(data.copy()))

    def forward(self, x):
        x = (
            (x @ self.w.flatten(0, 1).T + self.b.flatten())
            .reshape(-1, 96, 3)
            .max(2)
            .values
        )
        mean = x.mean(1, keepdim=True)
        var = ((x - mean) ** 2).mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-8) * self.gain + self.bias


class POSTagger(torch.nn.Module):
    """Thinc MultiHashEmbed + MaxoutWindowEncoder + linear tagger, one doc/run.

    Integer hash buckets stay outside ONNX (the Go code computes them). The
    trained en_core_web_sm weights are copied unchanged; nothing is retrained.
    """

    def __init__(self, nlp):
        super().__init__()
        nodes = list(nlp.get_pipe("tok2vec").model.walk())
        self.hashes = [m for m in nodes if m.name == "hashembed"]
        for i, m in enumerate(self.hashes):
            self.register_buffer(
                f"embedding{i}", torch.from_numpy(m.get_param("E").copy())
            )
        self.blocks = torch.nn.ModuleList(
            [
                Maxout(a, b)
                for a, b in zip(
                    [m for m in nodes if m.name == "maxout"],
                    [m for m in nodes if m.name == "layernorm"],
                    strict=True,
                )
            ]
        )
        linear = nlp.get_pipe("tagger").model.get_ref("softmax")
        self.register_buffer("w", torch.from_numpy(linear.get_param("W").copy()))
        self.register_buffer("b", torch.from_numpy(linear.get_param("b").copy()))

    def forward(self, buckets):
        # Sum in the same order as Thinc gather_add rather than a tree reduction.
        embedded = []
        for i in range(6):
            e = getattr(self, f"embedding{i}")[buckets[:, i, :]]
            embedded.append(((e[:, 0] + e[:, 1]) + e[:, 2]) + e[:, 3])
        x = self.blocks[0](torch.cat(embedded, 1))
        x = torch.nn.functional.pad(x, (0, 0, 4, 4))
        for block in self.blocks[1:]:
            padded = torch.nn.functional.pad(x, (0, 0, 1, 1))
            window = torch.cat([padded[:-2], padded[1:-1], padded[2:]], 1)
            x = x + block(window)
        vectors = x[4:-4]
        return vectors, vectors @ self.w.T + self.b


def buckets_for(nlp, doc):
    model = nlp.get_pipe("tok2vec").model
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"]
    features = doc.to_array(attrs)
    hashes = [m for m in model.walk() if m.name == "hashembed"]
    return np.stack(
        [
            m.ops.hash(np.ascontiguousarray(features[:, i]), m.attrs["seed"])
            % m.get_param("E").shape[0]
            for i, m in enumerate(hashes)
        ],
        axis=1,
    ).astype(np.int64)


def export_neural(cache, out):
    target = out / "fallback/neural/models"
    target.mkdir(parents=True, exist_ok=True)
    for dialect in ("us", "gb"):
        source = cache / dialect
        model = BartForConditionalGeneration.from_pretrained(
            source, local_files_only=True, attn_implementation="eager"
        ).eval()
        ids = torch.tensor([[1, 15, 20, 2]], dtype=torch.int64)
        hidden = Encoder(model)(ids).detach()
        export(
            Encoder(model),
            (ids,),
            target / f"{dialect}-encoder.onnx",
            ["input_ids"],
            ["hidden"],
            {"input_ids": {1: "source"}, "hidden": {1: "source"}},
        )
        export(
            Decoder(model),
            (torch.tensor([[1, 4]]), hidden),
            target / f"{dialect}-decoder.onnx",
            ["decoder_ids", "hidden"],
            ["logits"],
            {"decoder_ids": {1: "target"}, "hidden": {1: "source"}},
        )
        config = json.loads((source / "config.json").read_text())
        write_json(
            target / f"{dialect}.json",
            {
                "graphemes": config["grapheme_chars"],
                "phonemes": config["phoneme_chars"],
                "max_positions": 64,
                # Transformers 4.51.3 adds the initial decoder token to the
                # default max_length of 20 when generating.
                "max_length": model.generation_config.max_length + 1,
                "bos": 1,
                "eos": 2,
                "unknown": 3,
                "forced_eos": 2,
            },
        )


def export_spacy(out):
    from spacy.lang.norm_exceptions import BASE_NORMS
    from spacy.symbols import IDS

    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    pos = POSTagger(nlp)
    buckets = buckets_for(nlp, nlp("Export the trained tagging model."))
    export(
        pos,
        (torch.from_numpy(buckets),),
        out / "models/pos.onnx",
        ["buckets"],
        ["vectors", "scores"],
        {name: {0: "tokens"} for name in ("buckets", "vectors", "scores")},
    )
    write_json(
        out / "models/pos.json",
        {
            "labels": list(nlp.get_pipe("tagger").labels),
            "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"],
            "seeds": [m.attrs["seed"] for m in pos.hashes],
            "rows": [m.get_param("E").shape[0] for m in pos.hashes],
        },
    )
    # The tokenizer's regexes need lookaround, so they are exported as data and
    # evaluated in Go with regexp2.
    tok = nlp.tokenizer
    write_json(
        out / "data/tokenizer.json",
        {
            "rules": tok.rules,
            "prefix": tok.prefix_search.__self__.pattern,
            "suffix": tok.suffix_search.__self__.pattern,
            "infix": tok.infix_finditer.__self__.pattern,
            "url": tok.url_match.__self__.pattern,
            "norms": dict(nlp.vocab.lookups.get_table("lexeme_norm")),
            "symbols": IDS,
            "base_norms": BASE_NORMS,
        },
    )


def export_data(cache, out):
    (out / "data").mkdir(parents=True, exist_ok=True)
    for name in ("us_gold", "us_silver", "gb_gold", "gb_silver"):
        (out / f"data/{name}.json").write_bytes((cache / f"misaki/{name}.json").read_bytes())
    # Python's isdigit includes non-decimal digits, unlike Go's unicode.IsDigit.
    write_json(
        out / "data/unicode.json",
        {
            "version": unicodedata.unidata_version,
            "digits": {
                str(i): unicodedata.digit(chr(i))
                for i in range(0x110000)
                if chr(i).isdigit()
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="directory to write the repository layout into")
    parser.add_argument("--cache", type=Path, default=Path(".cache"), help="download cache")
    parser.add_argument("--check", action="store_true", help="compare outputs with the committed files")
    args = parser.parse_args()
    if unicodedata.unidata_version != "15.0.0":
        sys.exit(f"Python {sys.version.split()[0]} has Unicode {unicodedata.unidata_version}; use Python 3.12 (Unicode 15.0.0)")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    fetch(args.cache)
    export_data(args.cache, args.output)
    export_spacy(args.output)
    export_neural(args.cache, args.output)
    print(f"Exported {len(OUTPUTS)} files to {args.output}")
    if args.check:
        differ = [p for p in OUTPUTS if sha256(args.output / p) != sha256(REPO / p)]
        for path in differ:
            print(f"differs from committed: {path}")
        if differ:
            sys.exit(1)
        print("All outputs match the committed files")


if __name__ == "__main__":
    main()
