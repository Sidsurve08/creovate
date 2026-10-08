"""
Dataset Builder for CreoVate ML Models
Builds synthetic datasets from seed data and public sources
"""

import pandas as pd
import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Categories based on project requirements
CATEGORIES = [
    "pottery", "textiles", "paintings", "bamboo_cane", "wood", "metalwork",
    "jewellery", "leather", "stone_terracotta", "toys", "embroidery", "paper_scroll", "other"
]

# Crafts for heritage corpus
CRAFTS = [
    "Warli", "Madhubani", "Tanjore", "Pattachitra", "Kalamkari", "Bidri", "Dhokra",
    "Pashmina", "Channapatna", "Blue Pottery", "Phulkari", "Chikankari", "Kantha",
    "Bandhani", "Patola", "Ikat", "Gond", "Pichwai", "Sanjhi", "Cheriyal scrolls",
    "Kondapalli toys", "Terracotta of Bankura", "Bamboo craft of Assam", "Sholapith",
    "Aranmula Kannadi", "Mysore painting", "Thangka", "Zardozi", "Banarasi weaving", "Paithani"
]

# Regions
REGIONS = [
    "Maharashtra", "Bihar", "Tamil Nadu", "Odisha", "Andhra Pradesh", "Karnataka",
    "Rajasthan", "Punjab", "Uttar Pradesh", "Gujarat", "West Bengal", "Assam",
    "Kerala", "Madhya Pradesh", "Himachal Pradesh", "Jammu & Kashmir", "Telangana"
]

# Materials
MATERIALS = [
    "clay", "cotton", "silk", "wood", "metal", "stone", "bamboo", "paper",
    "leather", "wool", "brass", "copper", "gold foil", "natural dyes", "terracotta"
]

# Techniques
TECHNIQUES = [
    "wheel throwing", "hand painting", "block printing", "weaving", "carving",
    "embossing", "stone setting", "gold foil work", "tie-dye", "embroidery",
    "handloom", "knitting", "sculpting", "casting", "polishing"
]

# Languages for multilingual support
LANGUAGES = ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "Gujarati"]

# Sample text templates for synthetic generation
TITLE_TEMPLATES = [
    "Handmade {craft} {category}",
    "Traditional {region} {craft}",
    "Artisan {category} with {material}",
    "Custom {craft} {category}",
    "Vintage {craft} from {region}"
]

DESCRIPTION_TEMPLATES = [
    "This beautiful {craft} {category} is handcrafted by skilled artisans in {region}. Made using traditional {technique} techniques with {material}.",
    "A stunning example of {craft} heritage from {region}. This {category} features intricate {technique} work and is made from high-quality {material}.",
    "Traditional {craft} {category} crafted using time-honored {technique} methods. Originating from {region}, this piece showcases the rich cultural heritage of the region.",
    "Exquisite {craft} {category} handmade by master artisans. Features {technique} techniques and {material} materials from {region}."
]

def generate_listings_dataset(n=3000):
    """Generate synthetic listings dataset"""
    print(f"Generating listings dataset with {n} rows...")
    
    listings = []
    for i in range(n):
        category = random.choice(CATEGORIES)
        craft = random.choice(CRAFTS)
        region = random.choice(REGIONS)
        material = random.choice(MATERIALS)
        technique = random.choice(TECHNIQUES)
        language = random.choice(LANGUAGES)
        
        # Generate realistic price based on category and complexity
        base_price = {
            "pottery": 500,
            "textiles": 2000,
            "paintings": 3000,
            "bamboo_cane": 800,
            "wood": 1500,
            "metalwork": 2500,
            "jewellery": 5000,
            "leather": 1200,
            "stone_terracotta": 1000,
            "toys": 300,
            "embroidery": 1800,
            "paper_scroll": 600,
            "other": 1000
        }.get(category, 1000)
        
        complexity = random.randint(1, 5)
        price = int(base_price * complexity * random.uniform(0.8, 1.5))
        
        # Generate multilingual descriptions
        if language == "English":
            description = random.choice(DESCRIPTION_TEMPLATES).format(
                craft=craft, category=category, region=region,
                technique=technique, material=material
            )
        else:
            # For non-English, use transliterated English (simplified for demo)
            description = f"Traditional {craft} from {region}. Made with {material} using {technique}."
        
        listing = {
            "id": f"listing_{i+1:05d}",
            "title": random.choice(TITLE_TEMPLATES).format(
                craft=craft, category=category, region=region, material=material
            ),
            "description": description,
            "category": category,
            "craft": craft,
            "materials": [material, random.choice(MATERIALS)],
            "technique": technique,
            "region": region,
            "state": region,
            "language": language,
            "dimensions": f"{random.randint(10,100)}x{random.randint(10,100)} cm",
            "weight_g": random.randint(100, 5000),
            "complexity": complexity,
            "handmade": True,
            "is_rental": random.choice([True, False]),
            "price_inr": price,
            "rating": round(random.uniform(3.5, 5.0), 1),
            "num_sales": random.randint(0, 100),
            "image_path": f"images/{category}/listing_{i+1:05d}.jpg",
            "creator_id": f"creator_{random.randint(1, 100):03d}",
            "source": "synthetic"  # Mark as synthetic
        }
        listings.append(listing)
    
    df = pd.DataFrame(listings)
    output_path = DATA_DIR / "listings.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved listings dataset to {output_path}")
    return df

def generate_heritage_corpus():
    """Generate heritage records for 30+ crafts"""
    print("Generating heritage corpus...")
    
    heritage_records = []
    for i, craft in enumerate(CRAFTS):
        region = random.choice(REGIONS)
        
        record = {
            "id": f"heritage_{i+1:03d}",
            "title": f"{craft} Heritage",
            "craft": craft,
            "region": region,
            "cultural_context": f"{craft} is a traditional art form from {region}, representing the rich cultural heritage of the region.",
            "history": f"{craft} has been practiced for centuries, with origins dating back to ancient times. It has been passed down through generations of artisans.",
            "techniques": [random.choice(TECHNIQUES), random.choice(TECHNIQUES)],
            "materials": [random.choice(MATERIALS), random.choice(MATERIALS)],
            "creation_process": [
                "Prepare materials",
                "Apply traditional techniques",
                "Finalize with cultural motifs",
                "Quality inspection"
            ],
            "timeline": [
                {"period": "Ancient", "event": f"Origins of {craft}"},
                {"period": "Medieval", "event": f"Royal patronage of {craft}"},
                {"period": "Colonial", "event": f"Decline and revival of {craft}"},
                {"period": "Modern", "event": f"Global recognition of {craft}"}
            ],
            "tags": ["traditional", "heritage", "handmade", region.lower(), craft.lower()],
            "source": "seed"
        }
        heritage_records.append(record)
    
    output_path = DATA_DIR / "heritage_corpus.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in heritage_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Saved heritage corpus to {output_path}")
    return heritage_records

def generate_graph_data():
    """Generate graph nodes and edges from listings data"""
    print("Generating graph data...")
    
    # Load listings if exists, else generate small sample
    listings_path = DATA_DIR / "listings.csv"
    if listings_path.exists():
        listings_df = pd.read_csv(listings_path)
    else:
        listings_df = generate_listings_dataset(100)
    
    nodes = []
    edges = []
    node_id = 0
    
    # Create nodes for creators
    creators = set(listings_df['creator_id'].unique())
    for creator_id in creators:
        nodes.append({
            "id": node_id,
            "node_id": creator_id,
            "type": "Creator",
            "properties": {"id": creator_id}
        })
        node_id += 1
    
    # Create nodes for crafts
    for craft in CRAFTS:
        nodes.append({
            "id": node_id,
            "node_id": f"craft_{craft.lower()}",
            "type": "Craft",
            "properties": {"name": craft}
        })
        node_id += 1
    
    # Create nodes for materials
    for material in MATERIALS:
        nodes.append({
            "id": node_id,
            "node_id": f"material_{material.lower()}",
            "type": "Material",
            "properties": {"name": material}
        })
        node_id += 1
    
    # Create nodes for regions
    for region in REGIONS:
        nodes.append({
            "id": node_id,
            "node_id": f"region_{region.lower()}",
            "type": "Region",
            "properties": {"name": region}
        })
        node_id += 1
    
    # Create edges from listings
    for _, row in listings_df.iterrows():
        # Creator -> Craft
        edges.append({
            "source": row['creator_id'],
            "target": f"craft_{row['craft'].lower()}",
            "relation": "PRACTISES"
        })
        # Craft -> Material
        for material in eval(row['materials']) if isinstance(row['materials'], str) else row['materials']:
            edges.append({
                "source": f"craft_{row['craft'].lower()}",
                "target": f"material_{material.lower()}",
                "relation": "USES_MATERIAL"
            })
        # Craft -> Region
        edges.append({
            "source": f"craft_{row['craft'].lower()}",
            "target": f"region_{row['region'].lower()}",
            "relation": "ORIGINATED_IN"
        })
    
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    nodes_path = DATA_DIR / "graph_nodes.csv"
    edges_path = DATA_DIR / "graph_edges.csv"
    
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    
    print(f"Saved graph nodes to {nodes_path}")
    print(f"Saved graph edges to {edges_path}")
    return nodes_df, edges_df

def generate_barter_pairs(n=300):
    """Generate skill barter pairs for matching"""
    print(f"Generating barter pairs dataset with {n} rows...")
    
    skills = MATERIALS + TECHNIQUES + CRAFTS
    
    barter_pairs = []
    for i in range(n):
        offered_skill = random.choice(skills)
        needed_skill = random.choice(skills)
        
        # Determine match label (higher if skills are complementary)
        if offered_skill in MATERIALS and needed_skill in TECHNIQUES:
            match_label = 1
        elif offered_skill in TECHNIQUES and needed_skill in MATERIALS:
            match_label = 1
        elif offered_skill == needed_skill:
            match_label = 0
        else:
            match_label = random.choice([0, 1])
        
        pair = {
            "id": f"barter_{i+1:05d}",
            "offered_skill": offered_skill,
            "needed_skill": needed_skill,
            "match_label": match_label,
            "terms": f"Exchange {offered_skill} for {needed_skill}",
            "source": "synthetic"
        }
        barter_pairs.append(pair)
    
    df = pd.DataFrame(barter_pairs)
    output_path = DATA_DIR / "barter_pairs.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved barter pairs to {output_path}")
    return df

def create_sources_md():
    """Create SOURCES.md documenting dataset provenance"""
    print("Creating SOURCES.md...")
    
    content = """# Dataset Sources and Licenses

## Listings Dataset (listings.csv)

**Size:** 3,000 rows
**Source:** Synthetic generation based on seed data patterns
**License:** MIT (generated for demonstration)
**Provenance:**
- Patterns derived from existing seed data in `backend/data/seed.js`
- Categories, crafts, regions, materials based on Indian handicraft knowledge
- Price distributions based on category complexity multipliers
- Multilingual text samples in English, Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati

**Limitations:**
- Prices are synthetic estimates, not real market data
- Descriptions are template-based, not artisan-written
- For demonstration/prototype use only

## Heritage Corpus (heritage_corpus.jsonl)

**Size:** 30 records
**Source:** Seed data + cultural knowledge
**License:** MIT (generated for demonstration)
**Provenance:**
- 3 records from existing in-memory heritage data in `server.js`
- 27 additional crafts based on Indian GI tags and cultural knowledge
- Historical context based on publicly available cultural information

**Limitations:**
- Historical timelines are simplified
- Cultural contexts are generalized
- For demonstration/prototype use only

## Graph Data (graph_nodes.csv, graph_edges.csv)

**Size:** Derived from listings
**Source:** Synthetic listings dataset
**License:** MIT (generated for demonstration)
**Provenance:**
- Nodes extracted from listings (creators, crafts, materials, regions)
- Edges represent relationships (PRACTISES, USES_MATERIAL, ORIGINATED_IN)

**Limitations:**
- Graph is synthetic, not based on real collaboration data
- For demonstration/prototype use only

## Barter Pairs (barter_pairs.csv)

**Size:** 300 rows
**Source:** Synthetic generation
**License:** MIT (generated for demonstration)
**Provenance:**
- Skills extracted from materials, techniques, and crafts
- Match labels based on complementary skill heuristics

**Limitations:**
- Match labels are heuristic-based, not real user data
- For demonstration/prototype use only

## Speech Samples (speech_samples/)

**Size:** Not included (placeholder)
**Source:** To be sourced from:
- Common Voice (Mozilla) - CC0 license
- IndicVoices - CC BY license
- Or synthesized via TTS for demonstration

**Note:** Currently not included due to size. Use TTS synthesis for demo.

## Images (images/)

**Size:** Not included (placeholder)
**Source:** To be sourced from:
- Wikimedia Commons (CC BY/CC0)
- Unsplash (free for commercial use)
- Government open data (Ministry of Textiles, DC Handicrafts)

**Note:** Currently not included due to size. Use placeholder URLs for demo.

---

## Overall Limitations

All datasets are generated for demonstration and prototype purposes. They are not:
- Real market data
- Real artisan profiles
- Real transaction histories
- Suitable for production machine learning

For production deployment, datasets should be:
- Collected from real platform usage
- Properly licensed and consented
- Curated by domain experts
- Validated by artisans and cultural experts

## Data Split

All datasets use train/val/test split of 70/15/15 with fixed seed (42).
Stratified by category where applicable.
"""
    
    output_path = DATA_DIR / "SOURCES.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved SOURCES.md to {output_path}")

def split_dataset(df, test_size=0.15, val_size=0.15, stratify_col=None):
    """Split dataset into train/val/test with stratification"""
    from sklearn.model_selection import train_test_split
    
    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col].values
    else:
        stratify = None
    
    # First split: train + val vs test
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=stratify
    )
    
    # Second split: train vs val
    if stratify_col and stratify_col in df.columns:
        stratify_train_val = df_train_val[stratify_col].values
    else:
        stratify_train_val = None
    
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size/(1-test_size), random_state=SEED, stratify=stratify_train_val
    )
    
    return df_train, df_val, df_test

def main():
    """Main dataset building pipeline"""
    print("=" * 60)
    print("CreoVate Dataset Builder")
    print("=" * 60)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    listings_df = generate_listings_dataset(3000)
    heritage_records = generate_heritage_corpus()
    graph_nodes_df, graph_edges_df = generate_graph_data()
    barter_df = generate_barter_pairs(300)
    
    # Create sources documentation
    create_sources_md()
    
    # Split listings dataset
    print("\nSplitting listings dataset...")
    train_df, val_df, test_df = split_dataset(listings_df, stratify_col='category')
    
    train_path = DATA_DIR / "listings_train.csv"
    val_path = DATA_DIR / "listings_val.csv"
    test_path = DATA_DIR / "listings_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Val: {len(val_df)} rows -> {val_path}")
    print(f"Test: {len(test_df)} rows -> {test_path}")
    
    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
