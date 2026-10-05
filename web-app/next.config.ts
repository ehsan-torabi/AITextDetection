import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Next 16 dev blocks static chunks for hosts outside this list; allow the
  // loopback IP form alongside the default localhost so both URL forms work.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
