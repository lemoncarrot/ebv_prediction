import h5py

with h5py.File('../02_uni2_embeddings/69-A4_150026.h5') as f:
    print("root keys: ", list(f.keys()))
    
    print(f['coords'][:10])
    
    
    