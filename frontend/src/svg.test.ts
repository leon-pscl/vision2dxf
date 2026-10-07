import { describe, expect, it } from "vitest";
import { sanitiseSvg } from "./svg";

describe("sanitiseSvg", () => {
  it("keeps a legitimate pattern path", () => {
    const svg =
      '<svg viewBox="0 0 10 10"><path d="M0,0 L10,10" class="seam" stroke="#D6336C"/></svg>';
    const out = sanitiseSvg(svg);
    expect(out).toContain("<svg");
    expect(out).toContain("<path");
    expect(out).toContain('d="M0,0 L10,10"');
  });

  it("drops script tags", () => {
    const svg = '<svg><script>fetch("http://evil")</script><path d="M0,0"/></svg>';
    expect(sanitiseSvg(svg)).not.toContain("script");
  });

  it("drops event handlers", () => {
    const svg = '<svg><path d="M0,0" onload="alert(1)"/></svg>';
    expect(sanitiseSvg(svg)).not.toContain("onload");
  });

  it("drops external references", () => {
    const svg = '<svg><image href="http://evil/x.png"/><use xlink:href="http://evil/y"/></svg>';
    const out = sanitiseSvg(svg);
    expect(out).not.toContain("http://evil");
  });

  it("drops javascript urls", () => {
    const svg = '<svg><path d="M0,0" fill="javascript:alert(1)"/></svg>';
    expect(sanitiseSvg(svg)).not.toContain("javascript:");
  });

  it("empties a style block that smuggles something", () => {
    const svg = '<svg><style>@import url(http://evil/x.css);</style></svg>';
    expect(sanitiseSvg(svg)).not.toContain("@import");
  });

  it("rejects non-svg input", () => {
    expect(sanitiseSvg("<p>hello</p>")).toBe("");
  });
});
