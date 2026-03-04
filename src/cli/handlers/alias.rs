use needle::{Database, Result};

use crate::cli::commands::AliasCommands;

pub fn alias_command(cmd: AliasCommands) -> Result<()> {
    match cmd {
        AliasCommands::Create {
            database,
            alias,
            collection,
        } => alias_create(&database, &alias, &collection),
        AliasCommands::Delete { database, alias } => alias_delete(&database, &alias),
        AliasCommands::List { database } => alias_list(&database),
        AliasCommands::Resolve { database, alias } => alias_resolve(&database, &alias),
        AliasCommands::Update {
            database,
            alias,
            collection,
        } => alias_update(&database, &alias, &collection),
    }
}

fn alias_create(path: &str, alias: &str, collection: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    db.create_alias(alias, collection)?;
    db.save()?;

    println!("Created alias '{}' -> '{}'", alias, collection);
    Ok(())
}

fn alias_delete(path: &str, alias: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    let deleted = db.delete_alias(alias)?;
    db.save()?;

    if deleted {
        println!("Deleted alias '{}'", alias);
    } else {
        println!("Alias '{}' not found", alias);
    }
    Ok(())
}

fn alias_list(path: &str) -> Result<()> {
    let db = Database::open(path)?;
    let aliases = db.list_aliases();

    if aliases.is_empty() {
        println!("No aliases defined.");
    } else {
        println!("Aliases:");
        println!("{:-<50}", "");
        println!("{:<25} {:<25}", "Alias", "Collection");
        println!("{:-<50}", "");
        for (alias, collection) in aliases {
            println!("{:<25} {:<25}", alias, collection);
        }
    }

    Ok(())
}

fn alias_resolve(path: &str, alias: &str) -> Result<()> {
    let db = Database::open(path)?;

    match db.get_canonical_name(alias) {
        Some(collection) => {
            println!("{}", collection);
        }
        None => {
            println!("Alias '{}' not found", alias);
        }
    }

    Ok(())
}

fn alias_update(path: &str, alias: &str, collection: &str) -> Result<()> {
    let mut db = Database::open(path)?;
    db.update_alias(alias, collection)?;
    db.save()?;

    println!("Updated alias '{}' -> '{}'", alias, collection);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db_with_collection() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle").to_str().unwrap().to_string();
        let mut db = Database::open(&path).unwrap();
        db.create_collection("docs", 128).unwrap();
        db.save().unwrap();
        (dir, path)
    }

    #[test]
    fn test_alias_create_and_resolve() {
        let (_dir, path) = setup_db_with_collection();
        assert!(alias_create(&path, "my_alias", "docs").is_ok());
        assert!(alias_resolve(&path, "my_alias").is_ok());
    }

    #[test]
    fn test_alias_create_nonexistent_collection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.needle");
        let path = path.to_str().unwrap();
        let _db = Database::open(path).unwrap();

        let result = alias_create(path, "a", "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_alias_delete_existing() {
        let (_dir, path) = setup_db_with_collection();
        alias_create(&path, "del_alias", "docs").unwrap();
        assert!(alias_delete(&path, "del_alias").is_ok());
    }

    #[test]
    fn test_alias_delete_nonexistent() {
        let (_dir, path) = setup_db_with_collection();
        assert!(alias_delete(&path, "no_such_alias").is_ok());
    }

    #[test]
    fn test_alias_list_empty() {
        let (_dir, path) = setup_db_with_collection();
        assert!(alias_list(&path).is_ok());
    }

    #[test]
    fn test_alias_list_with_aliases() {
        let (_dir, path) = setup_db_with_collection();
        alias_create(&path, "a1", "docs").unwrap();
        assert!(alias_list(&path).is_ok());
    }

    #[test]
    fn test_alias_resolve_nonexistent() {
        let (_dir, path) = setup_db_with_collection();
        assert!(alias_resolve(&path, "no_alias").is_ok());
    }

    #[test]
    fn test_alias_update() {
        let (_dir, path) = setup_db_with_collection();
        alias_create(&path, "upd_alias", "docs").unwrap();

        let mut db = Database::open(&path).unwrap();
        db.create_collection("docs2", 64).unwrap();
        db.save().unwrap();

        assert!(alias_update(&path, "upd_alias", "docs2").is_ok());
    }

    #[test]
    fn test_alias_update_nonexistent() {
        let (_dir, path) = setup_db_with_collection();
        let result = alias_update(&path, "no_alias", "docs");
        assert!(result.is_err());
    }
}
