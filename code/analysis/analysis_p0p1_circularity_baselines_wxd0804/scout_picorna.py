#!/usr/bin/env python3
from pathlib import Path
import os
p = Path("/home/wangxindi/evo/evo_data/ig_species_wxd0729/Rhinovirus_C")
print("exists", p.exists())
if p.exists():
    for root, dirs, files in os.walk(p):
        print(root, "nfiles", len(files), "sample", files[:15])
        if len(files) > 0:
            break
mf = Path("/home/wangxindi/evo/evo_data/ig_multi_family_wxd0728/Picornaviridae")
print("mf", mf.exists())
if mf.exists():
    for root, dirs, files in os.walk(mf):
        print(root, "dirs", dirs[:10], "nfiles", len(files), "sample", files[:15])
        break
