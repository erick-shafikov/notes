```html
<section class="scrollbar">
  in faucibus orci luctus et ultrices posuere cubilia curae; Fusce non eleifend
  ipsum. Class aptent Nulla finibus tempor dignissim. in faucibus orci luctus et
  ultrices posuere cubilia curae; Fusce non eleifend ipsum. Class aptent Nulla
  finibus tempor dignissim. in faucibus orci luctus et ultrices posuere cubilia
  curae; Fusce non eleifend ipsum. Class aptent Nulla finibus tempor dignissim.
  in faucibus orci luctus et ultrices posuere cubilia curae; Fusce non eleifend
  ipsum. Class aptent Nulla finibus tempor dignissim. in faucibus orci luctus et
  ultrices posuere cubilia curae; Fusce non eleifend ipsum. Class aptent Nulla
  finibus tempor dignissim.
  <div class="bars">
    <div class="bar"></div>
  </div>
</section>
```

```scss
@property --start {
  syntax: "<number>";
  inherits: true;
  initial-value: 0;
}

@property --end {
  syntax: "<number>";
  inherits: true;
  initial-value: 0;
}

@keyframes start {
  from {
    --start: 1;
  }
  to {
    --start: 0;
  }
}

@keyframes end {
  from {
    --end: 0;
  }
  to {
    --end: 1;
  }
}

@keyframes bar {
  from,
  to {
    --opacity: 1;
    --events: auto;
  }
}

.scrollbar {
  scrollbar-width: none;
  scroll-timeline: --scrollbar;
  timeline-scope: --scrollbar;
  anchor-scope: --scrollbar;
  anchor-name: --scrollbar;
}

.bars {
  container-type: size;
  position: absolute;
  position-anchor: --scrollbar;
  inset: anchor(top) anchor(right) anchor(bottom) anchor(left);
  pointer-events: none;
}

.bar {
  min-block-size: 1.5rem;
  inset-block-start: min(calc(100% - var(--start) * 100%), calc(100% - 1rem));
  inset-block-end: calc(100% - var(--end) * 100%);
  opacity: var(--opacity, 0);
  pointer-events: var(--events, none);
  user-select: none;
  touch-action: none;

  animation-name: start, end, bar;
  animation-timing-function: linear;
  animation-timeline: --scrollbar;
  animation-range:
    0% calc(100% + 100cqb),
    calc(0% - 100cqb) 100%;
  animation-fill-mode: both;
}

/* Unrelated styles */
.scrollbar {
  width: 15rem;
  height: 15rem;
  margin: 1rem;
  padding: 1rem;
  scroll-padding: 1rem;
  box-sizing: border-box;
  font-family: sans-serif;
  background: whitesmoke;
  overflow: auto;
  resize: vertical;
}

.bars {
  border: 0.25rem solid transparent;
}

.bar {
  position: absolute;
  background: brown;
  inline-size: 0.5rem;
  border-radius: 1rem;
  inset-inline-end: 0;
  cursor: pointer;
}
```

```js
function wire(scrollbar) {
  scrollbar.querySelectorAll(".bar").forEach((bar) =>
    bar.addEventListener("pointerdown", ({ currentTarget, pageY, pageX }) => {
      const {
        scrollTop,
        scrollLeft,
        offsetHeight,
        offsetWidth,
        scrollHeight,
        scrollWidth,
      } = scrollbar;
      const rtl = scrollbar.matches('[dir="rtl"] :scope') ? -1 : 1;
      const vertical =
        currentTarget.parentElement.firstElementChild === currentTarget;
      const move = (e) => {
        const top = vertical
          ? scrollTop + ((e.pageY - pageY) / offsetHeight) * scrollHeight
          : scrollTop;
        const left = vertical
          ? scrollLeft
          : scrollLeft +
            ((rtl * (e.pageX - pageX)) / offsetWidth) * scrollWidth;

        scrollbar.scrollTo({ top, left, behavior: "instant" });
      };

      const end = () => {
        document.removeEventListener("pointermove", move);
        document.removeEventListener("pointerup", end);
        document.removeEventListener("pointercancel", end);
      };

      document.addEventListener("pointermove", move);
      document.addEventListener("pointerup", end);
      document.addEventListener("pointercancel", end);
    }),
  );
}

wire(document.querySelector(".scrollbar"));
```
