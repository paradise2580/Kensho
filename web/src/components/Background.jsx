/**
 * Purely decorative: three blurred colour fields and a masked grid.
 * `aria-hidden` and `pointer-events: none` keep it out of the accessibility
 * tree and out of the way of every control above it. It respects
 * `prefers-reduced-motion` in CSS rather than here.
 */
export default function Background() {
  return (
    <div className="bg" aria-hidden="true">
      <div className="bg-grid" />
      <div className="bg-orb bg-orb--a" />
      <div className="bg-orb bg-orb--b" />
      <div className="bg-orb bg-orb--c" />
    </div>
  )
}
