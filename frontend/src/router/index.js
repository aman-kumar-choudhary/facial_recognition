import { createRouter, createWebHistory } from 'vue-router'
import RegisterView from '../views/RegisterView.vue'
import AuthenticateView from '../views/AuthenticateView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/register' },
    { path: '/register', name: 'register', component: RegisterView },
    { path: '/verify', name: 'verify', component: AuthenticateView },
  ],
})

export default router
