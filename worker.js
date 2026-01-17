const params = new URLSearchParams(self.location.search)
const scriptName = params.get("script") || "./ahmetkayaify.js"

try {
  const ahmetKayaifyModule = await import(scriptName)
  const wasmName = scriptName.replace(".js", "_bg.wasm")

  await ahmetKayaifyModule.default(wasmName)
} catch (e) {
  console.error("worker failed to initialize:", e)
  throw e
}
