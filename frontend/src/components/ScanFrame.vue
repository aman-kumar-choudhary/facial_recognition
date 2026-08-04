<script setup>
// The signature element of the product: a viewfinder that visually
// communicates pipeline state (idle / scanning / success / rejected)
// through color and a sweeping scan line, echoing what a real
// biometric sensor overlay looks like.
defineProps({
  state: {
    type: String,
    default: 'no-face', // no-face | misaligned | ready | scanning | success | danger
  },
})
</script>

<template>
  <div class="scan-frame" :class="state">
    <span class="corner tl"></span>
    <span class="corner tr"></span>
    <span class="corner bl"></span>
    <span class="corner br"></span>
    <div v-if="state === 'ready' || state === 'scanning'" class="sweep"></div>
    <div class="oval-guide"></div>
    <slot />
  </div>
</template>

<style scoped>
.scan-frame {
  position: relative;
  width: min(100%, 320px);
  aspect-ratio: 3 / 4;
  margin: 0 auto;
  border-radius: 50%;
  overflow: hidden;
  background: #05070a;
  border: 2px solid rgba(204, 216, 226, 0.22);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

.corner {
  display: none;
}

.oval-guide {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 76%;
  height: 86%;
  transform: translate(-50%, -50%);
  border: 2px dashed rgba(199, 212, 223, 0.28);
  border-radius: 50%;
  z-index: 1;
  pointer-events: none;
}

.sweep {
  position: absolute;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  box-shadow: 0 0 14px var(--accent);
  animation: sweep-move 1.6s ease-in-out infinite;
  z-index: 2;
}

@keyframes sweep-move {
  0% {
    top: 8%;
  }
  50% {
    top: 92%;
  }
  100% {
    top: 8%;
  }
}

/* State color mapping */
.ready .corner,
.scanning .corner,
.ready .oval-guide,
.scanning .oval-guide {
  border-color: var(--success);
}
.ready,
.scanning {
  border-color: var(--success-dim);
  box-shadow: 0 0 22px rgba(61, 220, 132, 0.18);
}

.success .corner {
  border-color: var(--success);
}
.success {
  border-color: var(--success-dim);
  box-shadow: 0 0 22px rgba(61, 220, 132, 0.18);
}

.danger .corner,
.misaligned .corner,
.danger .oval-guide,
.misaligned .oval-guide {
  border-color: var(--danger);
}
.danger,
.misaligned {
  border-color: var(--danger-dim);
  box-shadow: 0 0 22px rgba(240, 97, 107, 0.16);
}

.no-face .corner,
.no-face .oval-guide {
  border-color: rgba(44, 73, 97, 0.9);
}
</style>
