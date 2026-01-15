# Python Installation

## Option 1: PyPI (recommended)

```bash
pip install needle-db
python -c "import needle; print('needle import ok')"
```

If your platform does not have a prebuilt wheel, pip will fall back to building from source.

## Option 2: Build from source

Requirements:
- Rust 1.85+
- Python 3.8+
- maturin

```bash
pip install maturin
maturin develop --features python
python -c "import needle; print('needle import ok')"
```

## Troubleshooting

- If the build fails, ensure `rustc --version` reports 1.85+.
- On macOS, you may need the Xcode Command Line Tools (`xcode-select --install`).
