import { createRouter, createWebHistory } from 'vue-router'
import RegisterView from '../views/RegisterView.vue'
import AuthenticateView from '../views/AuthenticateView.vue'
import ManagementView from '../views/ManagementView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/register' },
    { path: '/register', name: 'register', component: RegisterView },
    { path: '/verify', name: 'verify', component: AuthenticateView },
    { path: '/manage', name: 'manage', component: ManagementView },
  ],
})

export default router
