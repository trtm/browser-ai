import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_ACTIONS === "true";

const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true,
  },
  basePath: isGitHubPages ? "/browser-ai" : "",
  assetPrefix: isGitHubPages ? "/browser-ai/" : "",
  env: {
    NEXT_PUBLIC_BASE_PATH: isGitHubPages ? "/browser-ai" : "",
  },
};

export default nextConfig;
