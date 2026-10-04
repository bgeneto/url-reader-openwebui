/**
 * Build-time patch for the compiled Puppeteer service.
 *
 * Why: PuppeteerControl launches Chrome with `--single-process`
 * (see backend/functions/src/services/puppeteer.ts). Modern Chrome no longer
 * supports single-process mode. The browser process starts and reports itself
 * ready, then dies on the first CDP target command:
 *
 *   [CHANGE_LOGGER_NAME] INFO: Browser launched: 18
 *   [CHANGE_LOGGER_NAME] WARN: Browser disconnected
 *   TargetCloseError: Protocol error (Target.createTarget): Target closed
 *
 * That error is emitted as an unhandled 'error' event on PuppeteerControl, so
 * the whole Node process exits, `restart: always` brings it back, and every
 * crawl answers "500 Internal server error" in about 0.3s.
 *
 * This script strips the flag from the compiled output. Run it from /app, after
 * `npm run build`.
 *
 * Usage: node patch-puppeteer.cjs [path-to-compiled-file]
 */

const fs = require('fs');
const path = require('path');

const FLAG = '--single-process';
const target = process.argv[2] || path.join('build', 'services', 'puppeteer.js');

function fail(message) {
    console.error(`patch-puppeteer: ${message}`);
    process.exit(1);
}

if (!fs.existsSync(target)) {
    fail(`${target} not found - run "npm run build" first`);
}

const original = fs.readFileSync(target, 'utf8');

if (!original.includes(FLAG)) {
    // Show what the launch args actually look like, so a shape change is
    // diagnosable straight from the build log.
    const lines = original
        .split('\n')
        .map((line, i) => [i + 1, line])
        .filter(([, line]) => line.includes('no-sandbox') || line.includes('launch(') || line.includes('args'));

    console.error('patch-puppeteer: relevant lines in the compiled output:');
    for (const [n, line] of lines) {
        console.error(`  ${n}: ${line.trim()}`);
    }

    fail(
        `${FLAG} not found in ${target}. Either the bug is already fixed upstream ` +
            '(remove this patch from Dockerfile.reader) or the code changed shape.'
    );
}

// Literal replacements, so no regex escaping is involved. Step 1 removes the
// flag together with a following comma (mid-array); step 2 removes it when it is
// the last element; step 3 tidies a leftover double space.
const patched = original
    .split(`'${FLAG}',`).join('')
    .split(`'${FLAG}'`).join('')
    .split(',  ]').join(', ]');

if (patched.includes(FLAG)) {
    fail(`${FLAG} still present after patching - unexpected source formatting`);
}

fs.writeFileSync(target, patched);

const occurrences = original.split(FLAG).length - 1;
console.log(`patch-puppeteer: removed ${FLAG} (${occurrences} occurrence(s)) from ${target}`);
