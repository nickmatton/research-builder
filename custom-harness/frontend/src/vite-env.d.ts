/// <reference types="vite/client" />

// Vite's ?url import suffix returns a string URL for the resolved asset.
declare module "*?url" {
  const src: string;
  export default src;
}
