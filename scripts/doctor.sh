#!/usr/bin/env bash
set -euo pipefail

fail=0
warn=0

check_cmd() {
  local name="$1"
  local cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "✓ $name: $($cmd --version 2>/dev/null | head -n 1)"
  else
    echo "✗ $name: missing"
    fail=1
  fi
}

check_optional() {
  local name="$1"
  local cmd="$2"
  local install_hint="$3"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "✓ $name: $($cmd --version 2>/dev/null | head -n 1)"
  else
    echo "· $name: not installed (optional — $install_hint)"
    warn=1
  fi
}

echo "Needle environment check"
echo "========================"
echo ""
echo "Required:"

check_cmd "Rust (cargo)" cargo
check_cmd "Rust (rustc)" rustc
check_cmd "Git" git
check_cmd "Curl" curl

# Rust version and toolchain check
if command -v rustc >/dev/null 2>&1; then
  rust_version=$(rustc --version | awk '{print $2}')
  if [[ "$rust_version" < "1.85" ]]; then
    echo "✗ Rust version $rust_version detected; need 1.85+"
    if command -v rustup >/dev/null 2>&1; then
      echo "  Fix: rustup install 1.85 && rustup default 1.85"
      echo "  Or just run 'cargo build' — rustup reads rust-toolchain.toml automatically"
    else
      echo "  Fix: install rustup from https://rustup.rs/ then run: rustup install 1.85"
    fi
    fail=1
  fi
fi

if command -v rustup >/dev/null 2>&1; then
  echo "✓ Rustup: $(rustup --version 2>/dev/null | head -n 1)"
  active_tc=$(rustup show active-toolchain 2>/dev/null | head -n 1)
  if echo "$active_tc" | grep -q "1.85"; then
    echo "✓ Active toolchain: $active_tc"
  else
    echo "· Active toolchain: $active_tc"
    echo "  Note: rust-toolchain.toml pins 1.85; cargo will use it automatically"
  fi
else
  echo "· Rustup: not found (recommended for toolchain management — https://rustup.rs/)"
fi

echo ""
echo "Optional:"

check_optional "Docker" docker "needed for container deployment"
check_optional "Python" python3 "pip install maturin for Python bindings"
check_optional "Node.js" node "needed for JS/WASM SDK"
check_optional "just" just "cargo install just (or use make instead)"
check_optional "pre-commit" pre-commit "pip install pre-commit"
check_optional "maturin" maturin "pip install maturin (for Python bindings)"

echo ""

if [ "$fail" -ne 0 ]; then
  echo "✗ Some required tools are missing. Fixes:"
  echo "  - Install Rust:  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  echo "  - Install Git:   https://git-scm.com/downloads"
  echo "  - Install Curl:  https://curl.se/download.html"
  exit 1
else
  echo "✓ Ready to build! Try: make quick"
fi
