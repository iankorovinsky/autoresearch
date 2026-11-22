import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The Claude Agent SDK locates and spawns its bundled Claude Code binary
  // with Node's module resolution, which bundling would break.
  serverExternalPackages: ["@anthropic-ai/claude-agent-sdk"],
};

export default nextConfig;
