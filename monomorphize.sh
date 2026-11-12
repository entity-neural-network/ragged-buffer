#!/bin/bash
# Generates the i64 and bool Python classes from src/monomorphs/f32.rs.

cp src/monomorphs/{f32,i64}.rs
sed -i s/F32/I64/g src/monomorphs/i64.rs
sed -i s/f32/i64/g src/monomorphs/i64.rs
cp src/monomorphs/{f32,bool}.rs
sed -i s/F32/Bool/g src/monomorphs/bool.rs
sed -i s/f32/bool/g src/monomorphs/bool.rs
# bool has no arithmetic: drop the code between the ARITHMETIC markers.
sed -i '/BEGIN ARITHMETIC/,/END ARITHMETIC/d' src/monomorphs/bool.rs
sed -i "/use crate::monomorphs::RaggedBufferI64;/d" src/monomorphs/i64.rs
