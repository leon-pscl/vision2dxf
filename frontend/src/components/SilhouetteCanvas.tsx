import type { GarmentType } from "../types";

/** Front-view figure, drawn in a 0..100 x 0..190 box. 1px pink stroke, no fill. */
const BODY =
  "M50 4 C44 4 40 9 40 15 C40 19 42 22 42 25 " +
  "C42 30 41 33 38 35 L26 40 C22 42 20 46 20 50 L18 62 " +
  "C17 68 15 74 13 80 C12 84 10 88 9 92 L7 100 C7 102 9 103 11 102 " +
  "L17 96 C18 95 19 96 19 98 L20 112 C20 116 21 120 23 124 " +
  "L26 152 C26 156 27 158 29 158 L36 158 C38 158 39 156 39 154 " +
  "L41 124 C42 120 43 118 45 116 C47 118 48 120 49 124 " +
  "L51 154 C51 156 52 158 54 158 L61 158 C63 158 64 156 64 152 " +
  "L67 124 C69 120 70 116 70 112 L71 98 C71 96 72 95 73 96 " +
  "L79 102 C81 103 83 102 83 100 L81 92 C80 88 78 84 77 80 " +
  "C75 74 73 68 72 62 L70 50 C70 46 68 42 64 40 L52 35 " +
  "C49 33 48 30 48 25 C48 22 50 19 50 15 C50 9 56 4 50 4 Z";

/** Garment outlines, same coordinate box as the body. */
const GARMENT: Record<GarmentType, string> = {
  // collar, placket, short sleeves
  polo_shirt:
    "M38 33 L50 39 L62 33 L70 37 L78 41 L72 52 L68 49 L68 84 " +
    "C63 86 37 86 32 84 L32 49 L28 52 L22 41 L30 37 Z",
  // collar, placket, long sleeves
  long_sleeve_polo:
    "M38 33 L50 39 L62 33 L70 37 L80 41 L88 76 L82 78 L73 48 " +
    "L68 46 L68 84 C63 86 37 86 32 84 L32 46 L27 48 L18 78 L12 76 " +
    "L20 41 L30 37 Z",
  // flat rib collar, short sleeves, no placket
  short_sleeve_polo:
    "M38 34 L50 39 L62 34 L70 38 L78 41 L73 51 L68 48 L68 84 " +
    "C63 86 37 86 32 84 L32 48 L27 51 L22 41 L30 38 Z",
  // waistband and straight legs
  slacks:
    "M33 84 L67 84 L69 108 L67 156 L57 156 L50 110 L43 156 " +
    "L33 156 L31 108 Z",
};

interface Props {
  garment?: GarmentType;
  showGarment?: boolean;
}

/**
 * The sample UI's figure, reused as the workspace canvas on steps 1 to 6.
 * The grid behind it is CSS, so this is only the line art.
 */
export function SilhouetteCanvas({ garment = "polo_shirt", showGarment = true }: Props) {
  return (
    <svg
      className="canvas-figure"
      viewBox="-8 -6 116 202"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={showGarment ? `Body silhouette wearing ${garment.replace(/_/g, " ")}` : "Body silhouette"}
    >
      {showGarment && (
        <path className="figure-garment" d={GARMENT[garment]} />
      )}
      <path className="figure-body" d={BODY} />
    </svg>
  );
}
