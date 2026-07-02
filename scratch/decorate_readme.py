import re

readme_path = r"C:\Users\ishan\Documents\Projects\Awesome-Data-Parallelism\README.md"

with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add banner at the top if not present
banner_html = '<p align="center">\n  <img src="assets/banner.svg" alt="Awesome Data Parallelism Banner" width="100%">\n</p>\n\n'
if "banner.svg" not in content:
    content = banner_html + content

# Replace standard title and headings with emoji versions
content = content.replace("# Awesome-Data-Parallelism", "# 🚀 Awesome Data Parallelism 🚀")
content = content.replace("## Data Parallelism in AI:", "## 📚 Data Parallelism in AI:")
content = content.replace("## 1. The Macro Chronological Evolution", "## 📅 1. The Macro Chronological Evolution")
content = content.replace("## 2. Core Functional & Architectural Variants", "## 🏗️ 2. Core Functional & Architectural Variants")
content = content.replace("## 3. Communication Operations & Latency Mechanics", "## ⚡ 3. Communication Operations & Latency Mechanics")
content = content.replace("## 4. Production Engineering Challenges & Hardware Solutions", "## 🛠️ 4. Production Engineering Challenges & Hardware Solutions")
content = content.replace("## 5. Frontier Real-World AI Infrastructure Applications", "## 🌐 5. Frontier Real-World AI Infrastructure Applications")
content = content.replace("## References", "## 📖 References")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Decorated README with banner and emojis successfully.")
