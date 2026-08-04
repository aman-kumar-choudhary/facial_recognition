<script setup>
import { RouterLink, RouterView, useRoute } from 'vue-router'
const route = useRoute()
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-name">Checkpoint</span>
      </div>
      <nav class="tabs">
        <RouterLink to="/register" class="tab" :class="{ active: route.name === 'register' }">
          <span class="tab-index">01</span> Register
        </RouterLink>
        <RouterLink to="/verify" class="tab" :class="{ active: route.name === 'verify' }">
          <span class="tab-index">02</span> Verify
        </RouterLink>
      </nav>
    </header>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 32px;
  border-bottom: 1px solid var(--border-soft);
  flex-wrap: wrap;
  gap: 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.brand-mark {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  background: var(--accent);
  box-shadow: 0 0 12px var(--accent);
}

.brand-name {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 1.05rem;
  letter-spacing: 0.02em;
}

.tabs {
  display: flex;
  gap: 4px;
  background: var(--panel);
  border: 1px solid var(--border-soft);
  border-radius: var(--radius-md);
  padding: 4px;
}

.tab {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: var(--radius-sm);
  text-decoration: none;
  color: var(--text-muted);
  font-size: 0.9rem;
  font-weight: 500;
  transition: color 0.15s ease, background 0.15s ease;
}

.tab-index {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-faint);
}

.tab.active {
  color: var(--text-primary);
  background: var(--panel-raised);
}

.tab.active .tab-index {
  color: var(--accent);
}

.tab:hover:not(.active) {
  color: var(--text-primary);
}

.content {
  flex: 1;
  display: flex;
  padding: 32px;
}

@media (max-width: 640px) {
  .topbar {
    padding: 16px 20px;
  }
  .content {
    padding: 20px;
  }
}
</style>
