# pip install requests python-dotenv
"""
NCAAB Shared Configuration
===========================
Constants shared across NCAAB data tools.
"""

SEASON = 2026  # ESPN uses the ending year
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ODDS_BASE = "https://api.the-odds-api.com"

RATE_LIMIT_SEC = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# ---------------------------------------------------------------------------
# The 68 confirmed tournament teams
# ---------------------------------------------------------------------------

TEAMS = [
    # East (16)
    "Duke", "UConn", "Michigan State", "Kansas", "St. John's",
    "Louisville", "UCLA", "Ohio State", "TCU", "UCF",
    "South Florida", "Northern Iowa", "CA Baptist",
    "North Dakota State", "Furman", "Siena",
    # South (15 + Lehigh/Prairie View First Four)
    "Florida", "Houston", "Illinois", "Nebraska", "Vanderbilt",
    "North Carolina", "Saint Mary's", "Clemson", "Iowa", "Texas A&M",
    "VCU", "McNeese", "Troy", "Penn", "Idaho", "Lehigh",
    "Prairie View",
    # Midwest (15 + SMU/Miami(OH) First Four + Howard/UMBC First Four)
    "Michigan", "Iowa State", "Virginia", "Alabama", "Texas Tech",
    "Tennessee", "Kentucky", "Georgia", "Saint Louis", "Santa Clara",
    "SMU", "Miami (OH)", "Akron", "Hofstra", "Wright State",
    "Tennessee State", "Howard", "UMBC",
    # West (16 — Texas won First Four vs NC State)
    "Arizona", "Purdue", "Gonzaga", "Arkansas", "Wisconsin", "BYU",
    "Miami", "Villanova", "Utah State", "Missouri", "Texas",
    "High Point", "Hawai'i", "Kennesaw State", "Queens",
    "Long Island University", "NC State",
]

# ---------------------------------------------------------------------------
# Conference map: canonical team name -> 2025-26 conference
# ---------------------------------------------------------------------------

CONFERENCE_MAP: dict[str, str] = {
    # SEC
    "Florida": "SEC", "Alabama": "SEC", "Kentucky": "SEC",
    "Tennessee": "SEC", "Georgia": "SEC", "Vanderbilt": "SEC",
    "Missouri": "SEC", "Texas A&M": "SEC", "Texas": "SEC",
    "Arkansas": "SEC",
    # Big Ten
    "Michigan": "Big Ten", "Illinois": "Big Ten", "Purdue": "Big Ten",
    "Ohio State": "Big Ten", "Iowa": "Big Ten", "Wisconsin": "Big Ten",
    "Nebraska": "Big Ten", "UCLA": "Big Ten", "Michigan State": "Big Ten",
    # Big 12
    "Arizona": "Big 12", "Houston": "Big 12", "Iowa State": "Big 12",
    "Kansas": "Big 12", "Texas Tech": "Big 12", "TCU": "Big 12",
    "BYU": "Big 12", "UCF": "Big 12",
    # ACC
    "Duke": "ACC", "North Carolina": "ACC", "Louisville": "ACC",
    "Clemson": "ACC", "Virginia": "ACC", "Miami": "ACC",
    "NC State": "ACC", "SMU": "ACC",
    # Big East
    "St. John's": "Big East", "UConn": "Big East",
    "Villanova": "Big East",
    # WCC
    "Gonzaga": "WCC", "Saint Mary's": "WCC", "Santa Clara": "WCC",
    # A-10
    "VCU": "A-10", "Saint Louis": "A-10",
    # Mountain West
    "Utah State": "Mountain West",
    # AAC
    "South Florida": "AAC",
    # MVC
    "Northern Iowa": "MVC",
    # MAC
    "Akron": "MAC", "Miami (OH)": "MAC",
    # WAC
    "CA Baptist": "WAC",
    # Big South
    "High Point": "Big South",
    # CAA
    "Hofstra": "CAA",
    # Sun Belt
    "Troy": "Sun Belt",
    # Ivy
    "Penn": "Ivy",
    # SoCon
    "Furman": "SoCon",
    # America East
    "UMBC": "America East",
    # SWAC
    "Howard": "SWAC", "Prairie View": "SWAC",
    # OVC
    "Tennessee State": "OVC",
    # NEC
    "Long Island University": "NEC",
    # Patriot
    "Lehigh": "Patriot",
    # Summit
    "North Dakota State": "Summit",
    # Big Sky
    "Idaho": "Big Sky",
    # Big West
    "Hawai'i": "Big West",
    # ASUN
    "Queens": "ASUN", "Kennesaw State": "ASUN",
    # Horizon
    "Wright State": "Horizon",
    # Southland
    "McNeese": "Southland",
    # MAAC
    "Siena": "MAAC",
}

# ---------------------------------------------------------------------------
# ESPN name aliases for team-ID resolution
# ---------------------------------------------------------------------------

ESPN_NAME_ALIASES: dict[str, list[str]] = {
    "St. John's":              ["St. John's", "St. John's Red Storm", "St. John's (NY)"],
    "Miami (OH)":              ["Miami (OH)", "Miami Ohio", "Miami RedHawks"],
    "Miami":                   ["Miami", "Miami Hurricanes", "Miami (FL)"],
    "UConn":                   ["UConn", "Connecticut", "Connecticut Huskies"],
    "UCF":                     ["UCF", "Central Florida"],
    "VCU":                     ["VCU", "Virginia Commonwealth"],
    "SMU":                     ["SMU", "Southern Methodist"],
    "BYU":                     ["BYU", "Brigham Young"],
    "UMBC":                    ["UMBC", "Maryland-Baltimore County"],
    "NC State":                ["NC State", "North Carolina State"],
    "Long Island University":  ["Long Island University", "LIU", "Long Island"],
    "McNeese":                 ["McNeese", "McNeese State"],
    "North Dakota State":      ["North Dakota State", "North Dakota St"],
    "Wright State":            ["Wright State", "Wright St"],
    "Tennessee State":         ["Tennessee State", "Tennessee St"],
    "South Florida":           ["South Florida", "USF"],
    "Queens":                  ["Queens", "Queens University", "Queens (NC)",
                                "Queens University Royals", "Queens NC"],
    "High Point":              ["High Point"],
    "TCU":                     ["TCU", "Texas Christian"],
    "Northern Iowa":           ["Northern Iowa", "UNI"],
    "Michigan State":          ["Michigan State"],
    "Iowa State":              ["Iowa State"],
    "Texas A&M":               ["Texas A&M"],
    "Saint Mary's":            ["Saint Mary's", "Saint Mary's (CA)"],
    "Saint Louis":             ["Saint Louis"],
    "Santa Clara":             ["Santa Clara"],
    "North Carolina":          ["North Carolina", "UNC"],
    "Ohio State":              ["Ohio State"],
    "Texas Tech":              ["Texas Tech"],
    "CA Baptist":              ["CA Baptist", "California Baptist", "Cal Baptist", "CBU"],
    "Hawai'i":                 ["Hawai'i", "Hawaii", "Hawaii Rainbow Warriors"],
    "Kennesaw State":          ["Kennesaw State", "Kennesaw St"],
    "Penn":                    ["Penn", "Pennsylvania"],
    "Prairie View":            ["Prairie View", "Prairie View A&M", "PVAMU"],
}

# ---------------------------------------------------------------------------
# Odds API name mapping: canonical -> Odds API long-form name
# ---------------------------------------------------------------------------

ODDS_NAME_MAP: dict[str, str] = {
    "Duke":                    "Duke Blue Devils",
    "Siena":                   "Siena Saints",
    "Georgia":                 "Georgia Bulldogs",
    "TCU":                     "TCU Horned Frogs",
    "Arkansas":                "Arkansas Razorbacks",
    "Purdue":                  "Purdue Boilermakers",
    "Northern Iowa":           "Northern Iowa Panthers",
    "Louisville":              "Louisville Cardinals",
    "Missouri":                "Missouri Tigers",
    "VCU":                     "VCU Rams",
    "Illinois":                "Illinois Fighting Illini",
    "North Dakota State":      "North Dakota St Bison",
    "Kentucky":                "Kentucky Wildcats",
    "Santa Clara":             "Santa Clara Broncos",
    "Iowa State":              "Iowa State Cyclones",
    "Tennessee State":         "Tennessee St Tigers",
    "Florida":                 "Florida Gators",
    "Lehigh":                  "Lehigh Mountain Hawks",
    "Ohio State":              "Ohio State Buckeyes",
    "Clemson":                 "Clemson Tigers",
    "St. John's":              "St. John's Red Storm",
    "McNeese":                 "McNeese Cowboys",
    "Texas Tech":              "Texas Tech Red Raiders",
    "North Carolina":          "North Carolina Tar Heels",
    "South Florida":           "South Florida Bulls",
    "Nebraska":                "Nebraska Cornhuskers",
    "Wright State":            "Wright St Raiders",
    "Saint Mary's":            "Saint Mary's Gaels",
    "Saint Louis":             "Saint Louis Billikens",
    "Houston":                 "Houston Cougars",
    "Furman":                  "Furman Paladins",
    "Michigan":                "Michigan Wolverines",
    "Howard":                  "Howard Bison",
    "Idaho":                   "Idaho Vandals",
    "Utah State":              "Utah State Aggies",
    "Iowa":                    "Iowa Hawkeyes",
    "Tennessee":               "Tennessee Volunteers",
    "Akron":                   "Akron Zips",
    "Kansas":                  "Kansas Jayhawks",
    "Wisconsin":               "Wisconsin Badgers",
    "Miami (OH)":              "Miami (OH) RedHawks",
    "Alabama":                 "Alabama Crimson Tide",
    "Troy":                    "Troy Trojans",
    "Miami":                   "Miami Hurricanes",
    "UCF":                     "UCF Knights",
    "UConn":                   "UConn Huskies",
    "UMBC":                    "UMBC Retrievers",
    "Arizona":                 "Arizona Wildcats",
    "Long Island University":  "LIU Sharks",
    "UCLA":                    "UCLA Bruins",
    "Texas A&M":               "Texas A&M Aggies",
    "Vanderbilt":              "Vanderbilt Commodores",
    "High Point":              "High Point Panthers",
    "Virginia":                "Virginia Cavaliers",
    "Hofstra":                 "Hofstra Pride",
    "BYU":                     "BYU Cougars",
    "Texas":                   "Texas Longhorns",
    "SMU":                     "SMU Mustangs",
    "Gonzaga":                 "Gonzaga Bulldogs",
    "Villanova":               "Villanova Wildcats",
    "NC State":                "NC State Wolfpack",
    "Michigan State":          "Michigan St Spartans",
    "Queens":                  "Queens University Royals",
    "CA Baptist":              "California Baptist Lancers",
    "Hawai'i":                 "Hawaii Rainbow Warriors",
    "Kennesaw State":          "Kennesaw St Owls",
    "Penn":                    "Penn Quakers",
    "Prairie View":            "Prairie View A&M Panthers",
}

# Hardcoded ESPN IDs for teams not reliably in the paginated /teams endpoint
HARDCODED_ESPN_IDS: dict[str, str] = {
    "Queens": "2511",
}
