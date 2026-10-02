import os
import requests
from PIL import Image
from io import BytesIO


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection_extra",
    "images"
)

API_URL = "https://commons.wikimedia.org/w/api.php"


# ============================================================
# SETTINGS
# ============================================================

TARGET_IMAGES = 500

# Searches designed to give pose variation
SEARCHES = [
    "dog side profile",
    "dog profile",
    "dog 3/4 view",
    "dog three quarter view",
    "dog head side view",
    "dog face side view",
    "dog head profile",
    "dog face profile",
    "dog looking sideways",
    "dog head turned",
    "dog face turned",
    "dog portrait side view",
    "dog portrait 3/4 view",
    "dog muzzle side view",
    "dog nose side view",
    "dog looking left",
    "dog looking right",
    "dog side face",
    "dog angled face",
    "dog head angled",
]


HEADERS = {
    "User-Agent": (
        "CanineNosePrintRecognition/1.0 "
        "(research dataset collection)"
    )
}


# ============================================================
# SEARCH WIKIMEDIA
# ============================================================

def search_commons(search_term, limit=500):

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 1200,
        "format": "json",
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        print(
            f"Search failed: {search_term}"
        )
        print(e)

        return []

    pages = (
        data
        .get("query", {})
        .get("pages", {})
    )

    results = []

    for page in pages.values():

        imageinfo = page.get(
            "imageinfo"
        )

        if not imageinfo:
            continue

        info = imageinfo[0]

        url = (
            info.get("thumburl")
            or info.get("url")
        )

        if url:

            results.append({
                "title": page.get(
                    "title",
                    ""
                ),
                "url": url
            })

    return results


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

def download_image(
    url,
    output_path
):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if not content_type.startswith(
            "image/"
        ):
            return False

        image = Image.open(
            BytesIO(response.content)
        )

        # Validate image
        image.verify()

        # Reopen after verify
        image = Image.open(
            BytesIO(response.content)
        )

        width, height = image.size

        # Reject tiny images
        if width < 300 or height < 300:
            return False

        # Convert everything to JPEG
        image = image.convert("RGB")

        image.save(
            output_path,
            "JPEG",
            quality=90
        )

        return True

    except Exception:
        return False


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("       DOWNLOADING DOG IMAGES FROM WIKIMEDIA")
    print("=" * 70)

    print()
    print(
        f"Target: {TARGET_IMAGES} images"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    downloaded = 0

    seen_urls = set()
    seen_titles = set()

    # --------------------------------------------------------
    # Search each pose category
    # --------------------------------------------------------

    for search_term in SEARCHES:

        if downloaded >= TARGET_IMAGES:
            break

        print()
        print("=" * 70)
        print(
            f"SEARCH: {search_term}"
        )
        print("=" * 70)

        results = search_commons(
            search_term,
            limit=500
        )

        print(
            f"Results found: {len(results)}"
        )

        for result in results:

            if downloaded >= TARGET_IMAGES:
                break

            url = result["url"]
            title = result["title"]

            # Avoid duplicates
            if url in seen_urls:
                continue

            if title in seen_titles:
                continue

            seen_urls.add(url)
            seen_titles.add(title)

            filename = (
                f"web_dog_"
                f"{downloaded + 1:04d}.jpg"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            success = download_image(
                url,
                output_path
            )

            if success:

                downloaded += 1

                print(
                    f"[{downloaded:04d}/"
                    f"{TARGET_IMAGES}] "
                    f"SAVED {filename}"
                )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("                 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Successfully downloaded: "
        f"{downloaded}"
    )

    print()
    print(
        f"Saved to:"
    )

    print(
        OUTPUT_DIR
    )

    if downloaded < TARGET_IMAGES:

        print()
        print(
            "Target was not reached."
        )

        print(
            "Add more search terms if necessary."
        )

    else:

        print()
        print(
            "500-image target reached."
        )


if __name__ == "__main__":
    main()