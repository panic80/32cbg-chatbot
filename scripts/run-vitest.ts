#!/usr/bin/env tsx

import { startVitest } from 'vitest/node';

async function main() {
  const cliFilters = process.argv.slice(2);

  const ctx = await startVitest('test', cliFilters, {
    watch: false,
    run: true,
    reporters: 'default',
    pool: 'forks',
    test: {
      pool: {
        forks: {
          singleFork: true,
          minForks: 1,
          maxForks: 1,
        },
      },
    },
  });

  const state = ctx.state;
  const failedFiles = state.getFailedFilepaths().length;
  const unhandledErrors = state.getUnhandledErrors().length;
  const timeoutCauses = state.getProcessTimeoutCauses().length;

  await ctx.close();

  if (failedFiles || unhandledErrors || timeoutCauses) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
