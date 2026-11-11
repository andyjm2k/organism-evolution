import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dest = join(here, "..", "public", "config");
mkdirSync(dest, { recursive: true });
copyFileSync(join(here, "..", "..", "config", "living-world-config.json"), join(dest, "living-world-config.json"));
copyFileSync(join(here, "..", "..", "config", "neat-config.ini"), join(dest, "neat-config.ini"));
