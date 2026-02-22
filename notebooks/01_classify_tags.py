"""
Classify all 886 WoB tags into entity types for the Cosmere knowledge graph.

Categories:
- character: Named individuals, spren with names, important beings
- world: Planets, cities, regions, geographic locations
- magic: Magic systems, mechanics, substances, abilities
- shard: Shards of Adonalsium and shard-related concepts
- concept: Cosmere-level theories, phenomena, organizations, races/species
- book: Books, series, sequels, adaptations
- meta: Writing advice, process, admin tags, non-Cosmere content
"""

import json
from collections import Counter
from pathlib import Path

# Load WoB data
wob_path = Path(__file__).parent.parent.parent / "words-of-brandon" / "wob_entries.json"
with open(wob_path) as f:
    entries = json.load(f)

# Get all tags with counts
all_tags = []
for e in entries:
    all_tags.extend(e["tags"])
tag_counts = Counter(all_tags)

# ── Classification map ──────────────────────────────────────────────────────

CHARACTERS = {
    # Major Cosmere characters
    "hoid", "kelsier", "kaladin", "vin", "dalinar", "shallan", "jasnah",
    "szeth", "sazed", "rashek", "elend", "lift", "wayne", "vivenna",
    "khriss", "vasher", "adolin", "taravangian", "spook", "marsh", "nazh",
    "waxillium", "tensoon", "demoux", "sylphrena", "lopen", "navani",
    "gavilar", "raoden", "sarene", "siri", "hrathen", "taln", "frost",
    "david charleston", "spensa", "renarin", "marasi", "lightsong", "denth",
    "zane", "elhokar", "steris", "rysn", "rock", "moash", "eshonai",
    "melaan", "tien", "iyatil", "galladon", "mraize", "sigzil", "felt",
    "alcatraz smedry", "breeze", "sadeas", "amaram", "shai", "susebron",
    "blushweaver", "teft", "venli", "paalm", "ranette", "llarimar",
    "baon", "sja-anat", "davriel", "hesina", "tonk fah", "gaz", "lirin",
    "telsin", "allrianne", "evi", "shashara", "arsteel", "yesteel",
    "rlain", "bluefingers", "pattern", "glys", "ishar", "nale", "shalash",
    "jewels", "mare", "clubs", "dockson", "axies", "hammond", "bastille",
    "megan", "pailiah", "kalak", "daorn", "gemmel", "bloody tan",
    "oroden", "cord", "adien", "nicki savage", "joel saxon",
    "jonathan phaedrus", "stephen leeds", "kaise", "straff",
    "aesudan", "mizzy", "skar", "helaran", "jerick", "vathi", "dusk",
    "el", "raboniel", "amaranta", "malli", "gretgor", "ym", "kwaan",
    "eddlin", "vaxilian", "goradel", "darro", "balat", "lin davar",
    "alendi", "dilaf", "ashravan", "graves", "mrall", "xisis", "rua",
    "attica smedry", "cody", "abraham", "lemmex", "tukks", "riino",
    "nalizar", "agent ving", "aradel", "zeek harbringer",
    "longshadow", "tearim", "melody muns", "wode", "reluur",
    "silence montane", "vendell", "sunraiser", "obliteration", "roshone",
    "gavinor", "melishi", "wyrn", "ba-ado-mishram", "stormfather",
    "nightwatcher", "nohadon", "wit", "thaidakar", "wax", "vo",
    "mayalaran", "chiri-chiri", "stick", "jorgen", "m-bot", "doomslug",
    "drehy", "liss", "design", "reya", "wyndle", "ivory", "tarah",
    "miles", "ash", "edgli", "tanavast", "leras", "ati", "bavadin",
    "rayse", "sibling", "jezrien", "tindwyl", "spook",
    "paalm", "waxillium", "elhokar",
    "conflux novels",  # Actually a character (Conflux the Epic) - but tagged oddly
    "gaz", "tia", "senna", "aslydin", "jaddeth",
    "darro", "alcatraz smedry", "david charleston", "mizzy",
    "dreamer", "sovereign", "god king",
    # Reckoners characters
    "steelheart epic", "calamity epic",
    # Named spren / cognitive entities
    "the traveler", "mist spirit",
    "raboniel", "el",
    "yelig-nar",
    # Heralds (some already listed)
    "nale", "jezrien", "ishar", "taln", "shalash", "pailiah", "kalak",
    "zeek harbinger",
}

WORLDS = {
    # Planets
    "roshar", "scadrial", "sel", "nalthis", "taldain", "yolen",
    "threnody", "ashyn", "braize", "first of the sun", "obrodai", "lumar",
    "canticle", "detritus",
    # Cities / Regions
    "elantris city", "luthadel", "urithiru", "silverlight", "shadesmar",
    "hallandren", "idris", "terris", "alethkar", "shinovar", "purelake",
    "shattered plains", "celebrant chain", "elendel", "arelon", "fjorden",
    "aimia", "pits of hathsin", "rose empire", "duladel", "elendel basin",
    "dawncities", "kelesina mansion", "lasting integrity", "horneater peaks",
    "pahn kahl", "rall elorim", "uli da", "patji", "jindo",
    "southern scadrial", "darkside",
    # Systems / cosmography
    "rosharan system", "scadrian system", "threnodite system",
    "nalthian system", "drominad system", "cosmere star chart",
    # Locations
    "tower of nebrask",
}

MAGIC = {
    # Metallic Arts
    "allomancy", "feruchemy", "hemalurgy", "compounding", "atium",
    "lerasium", "aluminum", "time bubbles", "god metals", "metallic arts",
    "allomantic bronze", "allomantic copper", "nicrosil", "ettmetal",
    "trellium", "malatium", "gold", "silver", "emotional allomancy",
    "mechanical metallic arts", "hemalurgy table", "allomancy table",
    "feruchemy chart", "savantism", "snapping", "unsealed metalminds",
    "unkeyed metalminds", "steel alphabet", "leecher", "reverse compounding",
    # Surgebinding / Rosharan
    "surgebinding", "shardblades", "shardplate", "nahel bond",
    "soulcasting", "fabrials", "honorblades", "lashing", "lightweaving",
    "voidbinding", "voidbinding chart", "resonances", "polestones",
    "highstorms", "stormlight", "oathgates", "gemhearts", "division",
    "regrowth", "stoneshaping", "tension", "immortal words", "oaths",
    "glyphs", "spanreeds", "gemstones", "pure tones", "voidlight",
    "lifelight", "acclivity stone",
    # Awakening / Nalthian
    "awakening", "biochromatic breath", "returned", "divine breath",
    "royal locks", "lifeless", "drabs", "drab", "color",
    "tears of edgli", "eye color", "hair color",
    # Selish
    "aondor", "forgery", "the dor", "dor", "aons", "dakhor",
    "bloodsealing", "soulstamps", "chayshan", "slatrification",
    "selish magic", "hion lines",
    # Other
    "sand mastery", "aviar", "aethers", "microkinesis",
    "smedry talents", "occulator's lenses", "chalklings",
    "rithmatics", "kite magic",
    "yolish lightweaving", "corrupted investiture",
    "taynix", "cytonics",
    # Substances / artifacts
    "nightblood", "azure's sword", "oathbringer sword",
    "taln's honorblade", "moon scepter", "vin's earring",
    "wax's earring", "the bands of mourning", "bands of mourning",
    "ire connection orb",
    "vyres knife", "vyre's knife", "shardbows",
    # Concepts that are primarily magical mechanics
    "old magic", "the thrill",
    "anti-adonalsium weapon",
}

SHARDS = {
    # Named Shards of Adonalsium
    "honor", "odium", "cultivation", "preservation", "ruin", "harmony",
    "autonomy", "dominion", "devotion", "endowment", "ambition",
    "whimsy", "invention", "mercy", "valor",
    # Shard-related
    "adonalsium", "shards", "shattering of adonalsium", "shattering",
    "vessels", "original vessels", "shardic intents", "shardic champions",
    "splintering", "splinters", "slivers", "avatars",
    "unknown shards", "survival shard", "shard-not-on-a-planet",
    "discord",
}

CONCEPTS = {
    # Cosmere theory
    "investiture", "cognitive realm", "spiritual realm", "physical realm",
    "realmatic theory", "cosmere", "worldhopping", "worldhoppers",
    "perpendicularities", "perpendicularity", "connection", "identity",
    "fortune", "intention", "innate investiture", "manifestations of investiture",
    "cognitive shadows", "cognitive entities", "cosmere healing",
    "cognitive anomaly", "spiritweb", "spiritual ideals", "spiritual dna",
    "the beyond", "god beyond", "focus", "ascension",
    "dawnshards",
    # Spren (as a category/concept)
    "spren", "voidspren", "mistspren", "lightspren", "sapient spren",
    "deadeyes", "third bondsmith spren",
    # Races / Species / Groups
    "kandra", "koloss", "koloss-blooded", "listeners", "singers",
    "parshendi", "seons", "skaze", "elantrians", "sleepless",
    "herdazians", "horneaters", "horneater", "unkalaki", "shin",
    "alethi", "aimians", "siah aimians", "aimian", "larkins",
    "chasmfiends", "greatshells", "thunderclasts", "ryshadium",
    "chulls", "santhids", "cusicesh", "cremling", "crem",
    "mistwraiths", "inquisitors", "skaa", "epics", "krell",
    "fused", "regals", "heralds", "stormstriders", "deepest ones",
    "cosmere dragons", "liveborn", "hoed",
    # Organizations
    "knights radiant", "ghostbloods", "seventeenth shard", "bridge four",
    "sons of honor", "skybreakers", "windrunners", "bondsmiths",
    "truthwatchers", "elsecallers", "stonewards", "lightweavers",
    "order of truthwatchers", "order of bondsmiths", "order of dustbringers",
    "order of skybreakers", "order of willshapers", "order of edgedancers",
    "order of stonewards", "squires",
    "steel ministry", "ire", "worldsingers", "worldbringers",
    "five scholars", "ones above", "the set", "secret societies",
    "oldbloods",
    # Religion / Culture
    "vorinism", "shu-dereth", "shu-korath", "shu-keseg", "jesker mysteries",
    "jesker", "austrism", "passions", "pathism", "trellism", "survivorism",
    "church of the survivor", "religion", "hierocracy", "court of gods",
    "safehands", "women's script", "womens script", "double eye of the almighty",
    "terris alphabet", "thaylen script",
    # History / Events
    "chronology", "cosmere sequence", "desolations", "recreance",
    "false desolation", "scouring of aimia", "manywar", "catacendre",
    "shin invasions", "timelines", "reod", "shaod",
    # In-world texts / artifacts
    "death rattles", "epigraphs", "the letters", "diagram",
    "era 2 broadsheets", "ars arcanum", "keteks", "dawnchant",
    "girl who looked up", "rhythms",
    "stormfather visions", "oathpact",
    "iriali long trail", "dawnshard novella",
    # Phenomena
    "mists", "mistsickness", "everstorm", "weeping",
    "red haze", "the evil", "withering", "trell",
    "shades", "shadowblaze",
    # Cosmere mechanics
    "allomantic ftl", "space travel",
    "cosmere artificial intelligence",
    "multiple nahel bond",
    "true bodies",
    "handwavium",
    # Places that are more conceptual
    "the worldspire",
}

BOOKS = {
    # Stormlight Archive
    "stormlight archive", "the way of kings", "words of radiance",
    "oathbringer", "rhythm of war", "stormlight 4", "stormlight 5",
    "stormlight 6", "stormlight 7", "stormlight 8", "stormlight 9",
    "stormlight 10", "stormlight archive arc 2", "stormlight archive arc 1",
    "way of kings prime", "the way of kings in-world",
    "edgedancer novella",
    # Mistborn
    "mistborn", "mistborn: the final empire", "mistborn the final empire",
    "the well of ascension", "well of ascension", "the hero of ages",
    "final empire", "the alloy of law", "shadows of self",
    "the lost metal", "the bands of mourning book",
    "mistborn era 1", "mistborn era 2", "mistborn era 3", "mistborn era 4",
    "mistborn series", "mistborn prime", "final empire prime",
    "mistborn: secret history", "mistborn secret history",
    "mistborn secret history 2", "mistborn: secret history 2",
    # Warbreaker / Elantris
    "warbreaker", "warbreaker sequel", "nightblood book",
    "elantris book", "elantris", "elantris sequel",
    "the hope of elantris",
    # Emperor's Soul
    "the emperor's soul", "emperor's soul film",
    "emperors soul sequel", "the emperor's soul sequel", "emperor's soul sequel",
    # Cosmere collections
    "arcanum unbounded", "arcanum unbounded 2",
    "sixth of the dusk", "sixth of the dusk sequel",
    "shadows for silence in the forests of hell",
    "the silence divine",
    "dawnshard novella",
    # White Sand
    "white sand", "white sand prose", "white sand prime",
    "white sand vol 2", "white sand vol 3",
    "white sand omnibus", "white sand vol. 1", "white sand vol 1",
    "darkside graphic novels",
    # Dragonsteel
    "dragonsteel series", "dragonsteel prime", "dragonsteel",
    "liar of partinel",
    # Reckoners
    "reckoners", "steelheart book", "firefight book", "calamity book",
    "firefight", "lux",
    # Skyward / Cytoverse
    "skyward book", "skyward series", "starsight book", "cytonic",
    "defiant", "cytoverse",
    # Alcatraz
    "alcatraz series", "alcatraz book 6", "alcatraz book 2", "alcatraz book 1",
    # Other Sanderson
    "children of the nameless", "the rithmatist", "rithmatist series",
    "perfect state", "snapshot", "legion", "legion 2", "legion 3",
    "legion series", "dark one", "infinity blade", "the original",
    "aether of night", "the aztlanian", "knight life",
    "defending elysium", "i hate dragons", "the king's necromancer",
    "hoid parallel novel", "lord mastrell book", "silverlight novella",
    "star's end", "firstborn", "songs of the dead",
    "the night brigade", "the scar", "climb the sky",
    "death by pizza", "soulburner", "boatload of mummies",
    "mizzy book", "conflux novels", "the dark talent",
    "sixth incarnation of pandora", "kingmaker",
    "lunamor novella", "horneater novella",
    "the eleventh metal", "shadows beneath",
    "allomancer jak and the pits of eltania", "allomancer jak",
    # Adaptations / Games
    "mistborn film", "stormlight film", "skyward film",
    "steelheart film", "alcatraz film", "snapshot film",
    "dark one tv series", "legion tv series", "dark one podcast",
    "wheel of time tv show",
    "stormlight board game", "reckoners board game",
    "call to adventure stormlight", "stormlight vr experience",
    "stormlight rpg", "mistborn adventure game",
    "mistborn: house war", "mistborn house war",
    "mistborn: birthright", "mistborn birthright",
    "rithmatist video game", "reckoners video game",
    "mistborn coins", "mistcloak", "mistcoat",
    "card game", "video game",
    "stormlight art book",
    # Non-Cosmere reference
    "wheel of time", "magic the gathering",
    # Iridescent tones
    "iridescent tones",
    "wax and wayne 4",
}

# ── Stragglers (caught by unclassified check) ──────────────────────────────
# Characters
CHARACTERS.update({
    "skai", "aona",  # Vessels of Devotion and Dominion
    "vax", "kenton", "testament", "boomslug",
    "nale's unnamed companion", "terriswoman worldhopper",
    "kandra worldhopper", "sunmaker", "vanrial", "medelantorius",
    "odium's champion", "the sibling",
    "king lopen the first of alethkar",
})

# Concepts
CONCEPTS.update({
    "unmade",  # Group of Odium's spren
    "sho del",  # Race/species on Yolen
    "futuresight", "life sense",  # Abilities/phenomena
    "twinborn", "parshendi-blood",  # Types of people
    "dawnsingers", "skeletals", "cryptics", "iriali",  # Races/groups
    "svrakiss", "delvers",  # Entities
    "shardworlds",  # Cosmere concept
    "oathstones",  # Cultural artifact
    "magic system",  # General concept
    "terris prophesies", "unmaking",
    "sixteen",  # The Sixteen (original Vessels)
    "the eyes",  # Phenomenon
    "tamu keks",  # Cultural concept
    "palanaeum",  # Library (more concept than location)
    "ais",  # Plural of AI
    "alcatraz dinosaurs",  # Concept in Alcatraz
    "curse of kind",  # Phenomenon
    "tellingdwar",  # Cultural concept
    "tarachin",  # Sport/game
})

# Magic items/artifacts
MAGIC.update({
    "gavilar's black sphere", "hoid's flute", "reya's tear",
    "soulstone", "taldain sand",
    "elsecalling",  # Surge
})

# Worlds
WORLDS.update({
    "hallendren",  # Alternate spelling of Hallandren
    "enefel",  # Location
    "aether planet",  # Unnamed aether world
    "damnation",  # Braize / conceptual location
    "irali",  # Alternate of Iriali
})

# Books
BOOKS.update({
    "apocalypse guard", "threnody novel", "mythwalker", "adamant",
    "rithmatist",  # When used as book tag
    "shash",  # In-world writing
    "tzai blows",  # In-world reference
})

META = {
    # RAFO / admin
    "rafo", "rafo-plus", "rafo-resolved", "rafo explanation",
    "needs attention", "review transcription", "update footnote",
    "needs pictures", "add image source", "pafo", "pafo-resolved",
    "fix snippets", "fix video", "needs audio", "check for personalization",
    "entry needs to be split",
    # Writing / craft
    "writing", "writing process", "writing advice", "writing philosophy",
    "revisions", "inspirations", "influences", "favorites",
    "adaptations", "worldbuilding", "worldbuilding books",
    "recommendations", "beta readers", "publishing",
    "annotations", "readings", "book covers",
    "audiobooks", "graphic audio", "graphic novel",
    "leatherbound editions", "merchandise", "upcoming works",
    "collaborations", "fandom", "cosplay", "pronounciations",
    "illustrations", "maps",
    # People (real-world)
    "brandon", "isaac stewart", "brandon's wiki", "brandons wiki",
    "brandalizing", "kate reading", "michael kramer", "michael whelan",
    "byu class", "writing excuses",
    "17th shard fansite", "words of brandon",
    "dragonsteel entertainment",
    # Meta / fun
    "silly", "good question", "who would win", "fourth wall",
    "fashion", "food", "linguistics", "physics", "astronomy",
    "culture", "biology", "language", "research", "religion",
    "names", "characters", "signings", "unpublished",
    "crossovers", "chronology",
    # Non-cosmere refs
    "reckoners-earth", "rithmatist-earth", "alcatraz-earth",
    "reckoners multiverse",
    # Other
    "lgbtq", "autism-spectrum", "phone company",
    "artificial intelligence",
    "heraldic symbolism", "shadesmar map",
    "skyward space ships",
    "julia set", "grand apparatus",
    "fain life", "mythos", "kaladin soundtrack",
    "jasnah_deleted_scene",
    "unity", "reason",  # These are too vague/meta
    "gamma readers",
    "skycolors",
    "truthless",
    "high imperial",
}

# ── Build classification ────────────────────────────────────────────────────

classifications = {}
classified_tags = set()

for tag in tag_counts:
    if tag in CHARACTERS:
        classifications[tag] = "character"
        classified_tags.add(tag)
    elif tag in WORLDS:
        classifications[tag] = "world"
        classified_tags.add(tag)
    elif tag in MAGIC:
        classifications[tag] = "magic"
        classified_tags.add(tag)
    elif tag in SHARDS:
        classifications[tag] = "shard"
        classified_tags.add(tag)
    elif tag in CONCEPTS:
        classifications[tag] = "concept"
        classified_tags.add(tag)
    elif tag in BOOKS:
        classifications[tag] = "book"
        classified_tags.add(tag)
    elif tag in META:
        classifications[tag] = "meta"
        classified_tags.add(tag)

# Find unclassified tags
unclassified = set(tag_counts.keys()) - classified_tags
if unclassified:
    print(f"\n⚠ {len(unclassified)} unclassified tags:")
    for tag in sorted(unclassified, key=lambda t: -tag_counts[t]):
        print(f"  {tag_counts[tag]}\t{tag}")

# ── Summary ─────────────────────────────────────────────────────────────────

type_counts = Counter(classifications.values())
print(f"\nClassification summary:")
for typ, count in type_counts.most_common():
    print(f"  {typ}: {count} tags")
print(f"  TOTAL classified: {len(classifications)}/{len(tag_counts)}")

# Count WoB entries covered by non-meta tags
entity_tags = {t for t, c in classifications.items() if c != "meta"}
entries_with_entities = sum(
    1 for e in entries
    if any(t in entity_tags for t in e["tags"])
)
print(f"\nWoB entries with at least one entity tag: {entries_with_entities}/{len(entries)} "
      f"({entries_with_entities/len(entries)*100:.1f}%)")

# ── Save ────────────────────────────────────────────────────────────────────

output_path = Path(__file__).parent.parent / "data" / "tag_classifications.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save as {tag: {type, count}} for easy consumption
output = {
    tag: {"type": classifications[tag], "count": tag_counts[tag]}
    for tag in sorted(classifications.keys())
}

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {output_path}")
