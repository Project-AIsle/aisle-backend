# Optional LLM helpers (not used by default). Left as a stub to match structure.
def suggest_relations_from_text(product_name: str) -> list[str]:
    # Placeholder heuristic
    pairs = {
        "macarrão": ["molho de tomate", "queijo ralado"],
        "pão": ["manteiga", "geleia"],
    }
    return pairs.get(product_name.lower(), [])
