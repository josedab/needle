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
