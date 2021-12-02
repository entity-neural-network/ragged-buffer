#!/bin/bash

cp src/monomorphs/{f32,i64}.rs
sed -i s/F32/I64/g src/monomorphs/i64.rs
sed -i s/f32/i64/g src/monomorphs/i64.rs
sed -i "/use crate::monomorphs::RaggedBufferI64;/d" src/monomorphs/i64.rs
