import { createRouter, createWebHistory } from 'vue-router'
import useEventBus from '@/composables/useEventBus'
const { emitEvent } = useEventBus();
import NProgress from 'nprogress';

// admin
import AdminMain from '@/components/admin/AdminMain'
import AdminDashboard from '@/components/admin/admin-dashboard/AdminDashboard'
import ManageManager from '@/components/admin/admin-dashboard/manage-manager/ManageManager'
import AdminLogin from '@/components/admin/auth/AdminLogin'

// user 
import UserMain from '@/components/user/UserMain'
import AccountSetting from '@/components/user/account-setting/AccountSetting'
import MemberAccount from '@/components/user/member-account/MemberAccount'
import ManageContent from '@/components/user/manage-content/ManageContent'
import ManageBroadcast from '@/components/user/manage-broadcast/ManageBroadcast'
import StatisticalChannel from '@/components/user/statistical-channel/StatisticalChannel'
import UserLogin from '@/components/user/auth/UserLogin'
import UserResetPassword from '@/components/user/auth/UserResetPassword'

// ai + xla 
import FlowerRecognition from '@/components/user/flower-recognition/FlowerRecognition'
import MultipleFlowerRecognition from '@/components/user/flower-recognition/MultipleFlowerRecognition'
import IPMultipleFlowerRecognition from '@/components/user/flower-recognition/IPMultipleFlowerRecognition'
import AlzheimersRecognition from '@/components/user/alzheimers-recognition/AlzheimersRecognition'

// geo
import MapManage from '@/components/user/geo/MapManage'
import ViewDetailMap from '@/components/user/geo/ViewDetailMap'
import ViewMap from '@/components/user/geo/ViewMap'

// pbl7 
import PaperSearch from '@/components/user/pbl7/PaperSearch'
import PaperStatistical from '@/components/user/pbl7/PaperStatistical'

// other 
import CommonNotFound from '@/components/common/CommonNotFound'

// middleware authUser
// const authUser = (to, from, next) => {
//     const user = localStorage.getItem('user');
//     if (user) next();
//     else {
//         next({ name: 'UserLogin' });
//         emitEvent('eventError', 'You need to login !');
//     }
// };

// middleware authAdmin
const authAdmin = (to, from, next) => {
    const admin = localStorage.getItem('admin');
    if (admin) next();
    else {
        next({ name: 'AdminLogin' });
        emitEvent('eventError', 'You need to login !');
    }
};

// check user logged 
const loggedUser = (to, from, next) => {
    const user = localStorage.getItem('user');
    if (user) next({ name: 'AccountSetting' });
    else next();
};

// check amdin logged 
const loggedAdmin = (to, from, next) => {
    const user = localStorage.getItem('admin');
    if (user) next({ name: 'ManageManager' });
    else next();
};

const routes = [

    { path: '/login', component: UserLogin, name: 'UserLogin', beforeEnter: loggedUser },
    { path: '/reset-password', component: UserResetPassword, name: 'UserResetPassword', beforeEnter: loggedUser },
    {
        path: '/',
        component: UserMain,
        name: 'UserMain',
        // beforeEnter: authUser,
        children: [
            { path: 'account-setting', name: 'AccountSetting', component: AccountSetting },
            { path: 'member-account', name: 'MemberAccount', component: MemberAccount },
            { path: 'manage-content', name: 'ManageContent', component: ManageContent },
            { path: 'manage-broadcast', name: 'ManageBroadcast', component: ManageBroadcast },
            { path: 'statistical-channel', name: 'StatisticalChannel', component: StatisticalChannel },
            // ai + xla 
            { path: 'flower-recognition', name: 'FlowerRecognition', component: FlowerRecognition },
            { path: 'flower-multiple-recognition', name: 'MultipleFlowerRecognition', component: MultipleFlowerRecognition },
            { path: 'image-processing-flowers-recognition', name: 'IPMultipleFlowerRecognition', component: IPMultipleFlowerRecognition},
            { path: 'alzheimers-recognition', name: 'AlzheimersRecognition', component: AlzheimersRecognition },
            // geo 
            { path: 'manage-map', name: 'MapManage', component: MapManage },
            { path: 'detail-map/:id', name: 'ViewDetailMap', component: ViewDetailMap },
            // pbl7 
            { path: 'paper-search', name: 'PaperSearch', component: PaperSearch },
            { path: 'paper-statistical', name: 'PaperStatistical', component: PaperStatistical },
        ]
    },
    { path: '/view-map', component: ViewMap, name: 'ViewMap'},
    {
        path: '/admin',
        component: AdminMain,
        name: 'AdminMain',
        children: [
            { path: 'login', name: 'AdminLogin', component: AdminLogin, beforeEnter: loggedAdmin },
            {
                path: '',
                name: 'AdminDashboard',
                component: AdminDashboard,
                beforeEnter: authAdmin,
                children: [{ path: 'manage-manager', name: 'ManageManager', component: ManageManager }]
            }
        ]
    },
    { path: '/:CommonNotFound(.*)*', component: CommonNotFound, name: 'CommonNotFound' }
];

const router = createRouter({
    history: createWebHistory(),
    base: process.env.BASE_URL,
    routes: routes
})

router.beforeResolve((to, from, next) => {
    // Nếu đây không phải là lần tải trang đầu tiên.
    if (to.name) {
        // Bắt đầu thanh tiến trình tuyến đường.
        NProgress.start()
    }
    next()
})

router.afterEach(() => {
    // Hoàn thành hoạt ảnh của thanh tiến trình tuyến đường.
    NProgress.done()
})

export default router