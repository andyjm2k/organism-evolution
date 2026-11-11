import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dest = join(here, "..", "public", "config");
const source = join(here, "..", "..", "config");
const files = ["living-world-config.json", "neat-config.ini"];

mkdirSync(dest, { recursive: true });

const parentHasConfigs = files.every((name) => existsSync(join(source, name)));
if (!parentHasConfigs) {
  const shipped = files.every((name) => existsSync(join(dest, name)));
  if (!shipped) {
    console.error("Config files are missing from public/config.");
    process.exit(1);
  }
  console.log("copy-config: using the copies already in public/config");
  process.exit(0);
}

for (const name of files) {
  copyFileSync(join(source, name), join(dest, name));
}
