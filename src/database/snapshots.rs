use super::Database;
use crate::collection::Collection;
use crate::error::{NeedleError, Result};

impl Database {
    /// Create a named snapshot of a collection.
    ///
    /// The snapshot captures the full state of the collection and stores it
    /// in the database. Use [`restore_snapshot`] to restore from a snapshot.
    pub fn create_snapshot(&self, collection: &str, snapshot_name: &str) -> Result<()> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let snapshot_data = coll.create_snapshot()?;
        drop(state);

        // Store snapshot as a special collection named __snapshot__{collection}__{name}
        let snapshot_key = format!("__snapshot__{collection}__{snapshot_name}");
        let mut state = self.state.write();
        let snapshot_coll: Collection = serde_json::from_slice(&snapshot_data)
            .map_err(|e| NeedleError::InvalidInput(format!("Snapshot serialization failed: {e}")))?;
        state.collections.insert(snapshot_key, snapshot_coll);
        drop(state);
        self.mark_modified();
        Ok(())
    }

    /// Restore a collection from a named snapshot.
    ///
    /// Replaces the collection's current state with the snapshot data.
    pub fn restore_snapshot(&self, collection: &str, snapshot_name: &str) -> Result<()> {
        let snapshot_key = format!("__snapshot__{collection}__{snapshot_name}");
        let state = self.state.read();
        let snapshot = state
            .collections
            .get(&snapshot_key)
            .ok_or_else(|| NeedleError::InvalidInput(format!("Snapshot '{snapshot_name}' not found for collection '{collection}'")))?;

        let snapshot_data = snapshot.create_snapshot()?;
        drop(state);

        let restored: Collection = serde_json::from_slice(&snapshot_data)
            .map_err(|e| NeedleError::InvalidInput(format!("Failed to restore snapshot: {e}")))?;

        let mut state = self.state.write();
        state.collections.insert(collection.to_string(), restored);
        drop(state);
        self.mark_modified();
        Ok(())
    }

    /// List all snapshots for a collection.
    pub fn list_snapshots(&self, collection: &str) -> Vec<String> {
        let prefix = format!("__snapshot__{collection}__");
        let state = self.state.read();
        state
            .collections
            .keys()
            .filter_map(|key| key.strip_prefix(&prefix).map(String::from))
            .collect()
    }
}
