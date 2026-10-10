"""Tile encoders. Each exposes an EncoderSpec and an `encode` taking the uint8
(B, tile_size, tile_size, 3) RGB batches that SlideReader.read_tile produces."""
