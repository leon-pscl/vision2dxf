/**
 * Sanitise a backend SVG string before it goes into innerHTML.
 *
 * The mock adapters generate our own SVG today, but phase 2 will hand us SVGs
 * derived from user images, and SVG is a scripting context. So strip anything
 * that can execute or fetch: script, event handlers, external references.
 */
const ALLOWED_TAGS = new Set([
  "svg", "g", "path", "line", "polyline", "polygon", "rect", "circle", "ellipse",
  "text", "tspan", "defs", "style", "title", "use", "marker",
]);

const ALLOWED_ATTRS = new Set([
  "d", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
  "width", "height", "viewbox", "preserveaspectratio", "points", "transform",
  "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity",
  "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "opacity",
  "class", "text-anchor", "dominant-baseline", "font-size", "font-family",
  "letter-spacing", "vector-effect", "id", "marker-end", "marker-start",
]);

export function sanitiseSvg(svg: string): string {
  if (typeof DOMParser === "undefined") return svg; // SSR or a non-DOM env
  const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
  if (doc.querySelector("parsererror")) return "";

  const walk = (node: Element) => {
    for (const child of Array.from(node.children)) {
      if (!ALLOWED_TAGS.has(child.tagName.toLowerCase())) {
        child.remove();
        continue;
      }
      for (const attr of Array.from(child.attributes)) {
        const name = attr.name.toLowerCase();
        const value = attr.value.trim().toLowerCase();
        const dangerous =
          name.startsWith("on") ||
          name === "href" ||
          name === "xlink:href" ||
          name === "src" ||
          value.startsWith("javascript:") ||
          value.startsWith("data:text/html") ||
          value.startsWith("url(http");
        if (dangerous || !ALLOWED_ATTRS.has(name)) {
          child.removeAttribute(attr.name);
        }
      }
      walk(child);
    }
  };

  const root = doc.documentElement;
  if (root.tagName.toLowerCase() !== "svg") return "";
  for (const attr of Array.from(root.attributes)) {
    if (!ALLOWED_ATTRS.has(attr.name.toLowerCase())) root.removeAttribute(attr.name);
  }
  walk(root);

  // a <style> block is the one place CSS can smuggle something, so keep only
  // plain declarations: no @import, no expression, no url()
  root.querySelectorAll("style").forEach((s) => {
    const css = s.textContent ?? "";
    if (/@import|expression\s*\(|url\s*\(|javascript:/i.test(css)) s.textContent = "";
  });

  return root.outerHTML;
}
