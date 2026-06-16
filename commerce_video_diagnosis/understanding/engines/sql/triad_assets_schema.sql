PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS product_master_snapshot (
  product_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_product_id TEXT NOT NULL,
  leaf_category_id TEXT NOT NULL,
  leaf_category_name TEXT NOT NULL,
  product_name TEXT NOT NULL,
  brand_name TEXT NOT NULL,
  shop_name TEXT NULL,
  brand_asset_level TEXT NOT NULL,
  price_band TEXT NOT NULL,
  price_source TEXT NOT NULL,
  financial_risk_level TEXT NOT NULL,
  core_jtbd TEXT NOT NULL,
  trust_barrier_level TEXT NOT NULL,
  cognitive_barrier_level TEXT NOT NULL,
  habit_switch_barrier_level TEXT NOT NULL,
  diagnosis_version TEXT NOT NULL,
  diagnosis_generated_at TEXT NOT NULL,
  snapshot_hash TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(source_product_id, snapshot_hash)
);

CREATE INDEX IF NOT EXISTS idx_product_latest
  ON product_master_snapshot(source_product_id, diagnosis_generated_at DESC);

CREATE TABLE IF NOT EXISTS video_blueprint_master (
  blueprint_id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  source_product_id TEXT NOT NULL,
  product_snapshot_id INTEGER NOT NULL,
  request_id TEXT NOT NULL,
  generator_version TEXT NOT NULL,
  workflow_version TEXT NOT NULL,
  storyboard_source TEXT NOT NULL,
  semantic_bundle_count INTEGER NOT NULL,
  primary_hec_json TEXT NOT NULL,
  secondary_effects_json TEXT NOT NULL,
  slider_signature_json TEXT NOT NULL,
  risk_flags_json TEXT NOT NULL,
  semantic_bundles_json TEXT NOT NULL,
  segment_to_bundle_map_json TEXT NOT NULL,
  bundle_to_segment_range_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(idempotency_key),
  FOREIGN KEY (product_snapshot_id) REFERENCES product_master_snapshot(product_snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_blueprint_video
  ON video_blueprint_master(video_id);
CREATE INDEX IF NOT EXISTS idx_blueprint_product
  ON video_blueprint_master(source_product_id);

CREATE TABLE IF NOT EXISTS video_segment_fact_table (
  segment_record_id TEXT PRIMARY KEY,
  blueprint_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  source_product_id TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  segment_order INTEGER NOT NULL,
  start_sec REAL NOT NULL,
  end_sec REAL NOT NULL,
  bundle_id TEXT NULL,
  shot_size TEXT NOT NULL,
  camera_movement TEXT NOT NULL,
  lighting_tone TEXT NOT NULL,
  visual_subject TEXT NOT NULL,
  key_objects_json TEXT NOT NULL,
  actions_json TEXT NOT NULL,
  ocr_facts_json TEXT NOT NULL,
  audio_facts_json TEXT NOT NULL,
  rhythm_facts_json TEXT NOT NULL,
  annotation_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(blueprint_id, segment_id),
  FOREIGN KEY (blueprint_id) REFERENCES video_blueprint_master(blueprint_id)
);

CREATE INDEX IF NOT EXISTS idx_segment_blueprint_order
  ON video_segment_fact_table(blueprint_id, segment_order);
CREATE INDEX IF NOT EXISTS idx_segment_video
  ON video_segment_fact_table(video_id);
CREATE INDEX IF NOT EXISTS idx_segment_product
  ON video_segment_fact_table(source_product_id);
