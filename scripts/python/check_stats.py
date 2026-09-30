import urllib.request
import json

url = "https://api.github.com/repos/aquamarine-hoshino170/C-Luminous.org/releases"
req = urllib.request.Request(
    url, 
    headers={"User-Agent": "Luminous-Tracker", "Accept": "application/vnd.github+json"}
)

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print("=== 📊 Luminous GitHub Release Statistics ===")
        if not data:
            print("No releases found yet.")
        for release in data:
            print(f"Tag: {release['tag_name']} | Name: {release['name']}")
            assets = release.get('assets', [])
            if not assets:
                print("  -> No binary assets uploaded yet.")
            for asset in assets:
                print(f"  📦 Asset: {asset['name']} | Downloads: {asset['download_count']}")
except Exception as e:
    print(f"Error fetching stats: {e}")
