# Compiling from src
1. Run `nvcc -ptx ./src/kernel.cu -o kernel.ptx`
2. Run `cargo test`, you may want to `cargo clean` occassionally. You can do a `cargo run`but there is no main code as it is all rust tests from lib.
