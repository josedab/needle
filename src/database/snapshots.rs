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

#[cfg(test)]
mod tests {
    use crate::Database;
    use crate::error::NeedleError;

    fn setup_db_with_data() -> Database {
        let db = Database::in_memory();
        db.create_collection("coll", 4).unwrap();
        {
            let coll = db.collection("coll").unwrap();
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
            coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        }
        db
    }

    #[test]
    fn test_create_and_list_snapshot() {
        let db = setup_db_with_data();
        db.create_snapshot("coll", "snap1").unwrap();
        let snaps = db.list_snapshots("coll");
        assert_eq!(snaps.len(), 1);
        assert!(snaps.contains(&"snap1".to_string()));
    }

    #[test]
    fn test_create_snapshot_missing_collection() {
        let db = Database::in_memory();
        let result = db.create_snapshot("bad", "snap1");
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    #[test]
    fn test_restore_snapshot() {
        let db = setup_db_with_data();
        db.create_snapshot("coll", "snap1").unwrap();

        // Delete a vector then restore
        {
            let coll = db.collection("coll").unwrap();
            coll.delete("v1").unwrap();
            assert!(coll.get("v1").is_none());
        }

        db.restore_snapshot("coll", "snap1").unwrap();
        let coll = db.collection("coll").unwrap();
        assert!(coll.get("v1").is_some());
    }

    #[test]
    fn test_restore_missing_snapshot() {
        let db = setup_db_with_data();
        let result = db.restore_snapshot("coll", "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_snapshots_empty() {
        let db = setup_db_with_data();
        assert!(db.list_snapshots("coll").is_empty());
    }

    #[test]
    fn test_multiple_snapshots() {
        let db = setup_db_with_data();
        db.create_snapshot("coll", "s1").unwrap();
        db.create_snapshot("coll", "s2").unwrap();
        let snaps = db.list_snapshots("coll");
        assert_eq!(snaps.len(), 2);
    }
}
