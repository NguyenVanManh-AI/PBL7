<template>
    <div id="main">
        <div>
            <h3 class="title-channel"><i class="fa fa-regular fa-image"></i> View Image Pandora 360 </h3>
        </div>
        <div class="mt-2">
            <div class="mt-3">
                <div class="mb-2">
                    <h2 class="color-title"> <span style="color: #007BFF;"><i class="fa-solid fa-map-location-dot"></i> {{ detailMap.address }} </span> <i class="fa-solid fa-circle"></i> <span style="color: #28A745;">[ {{ detailMap.coordinates[0] }} , {{ detailMap.coordinates[1] }} ]</span> </h2> 
                </div>
                <div id="pandora">
                    <div ref="threejsContainer" class="threejs-container"></div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import UserRequest from '@/restful/UserRequest';
import config from '@/config';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import useEventBus from '@/composables/useEventBus'
const { emitEvent } = useEventBus();

export default {
    name: "ViewDetailMap",
    data() {
        return {
            renderer: null,
            scene: null,
            camera: null,
            cubeCamera: null,
            lightProbe: null,
            config: config,
            detailMap: {
                id: '',
                address: '',
                coordinates: [],
                imagefull: null,
                px: null,
                nx: null,
                py: null,
                ny: null,
                pz: null,
                nz: null,
                created_at: null,
                updated_at: null,
            },
        }
    },
    mounted() {
        emitEvent('eventTitleHeader', 'Channel Manages Maps');
        document.title = "View Detail Maps | Geomodelling";
        const id = this.$route.params.id;
        this.fetchMapDetail(id);
        window.addEventListener('resize', this.onWindowResize);
    },
    beforeUnmount() {
        window.removeEventListener('resize', this.onWindowResize);
    },
    methods: {
        async fetchMapDetail(id) {
            try {
                const { data, messages } = await UserRequest.get(`map/detail/${id}`);
                this.detailMap = data ; 
                const urls = [
                    data.px,
                    data.nx,
                    data.py,
                    data.ny,
                    data.pz,
                    data.nz,
                ];
                this.init(urls);
                emitEvent('eventSuccess', messages[0]);
            } catch (error) {
                if (error.messages) {
                    emitEvent('eventError', error.messages[0]);
                    console.error(error);
                }
            }
        },
        init(urls) {
            const container = this.$refs.threejsContainer;
            // Renderer
            this.renderer = new THREE.WebGLRenderer({ antialias: true });
            this.renderer.setPixelRatio(window.devicePixelRatio);
            this.renderer.setSize(container.clientWidth, window.innerHeight);
            container.appendChild(this.renderer.domElement);

            // Scene
            this.scene = new THREE.Scene();

            // Camera
            this.camera = new THREE.PerspectiveCamera(50, container.clientWidth / window.innerHeight, 1, 1000);
            this.camera.position.set(0, 0, 30);

            // Cube Camera
            const cubeRenderTarget = new THREE.WebGLCubeRenderTarget(256);
            this.cubeCamera = new THREE.CubeCamera(1, 1000, cubeRenderTarget);

            // Controls
            const controls = new OrbitControls(this.camera, this.renderer.domElement);
            controls.addEventListener('change', this.renderScene);
            controls.minDistance = 10;
            controls.maxDistance = 50;
            controls.enablePan = false;

            // Light Probe
            this.lightProbe = new THREE.LightProbe();
            this.scene.add(this.lightProbe);
            new THREE.CubeTextureLoader().load(urls, (cubeTexture) => {
                this.scene.background = cubeTexture;
                this.cubeCamera.update(this.renderer, this.scene);
                this.renderScene();
            });
        },
        onWindowResize() {
            const container = this.$refs.threejsContainer;
            this.renderer.setSize(container.clientWidth, window.innerHeight);
            this.camera.aspect = container.clientWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderScene();
        },
        renderScene() {
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }
        },
    }
}
</script>


<style scoped>

#pandora {
    border: 1px solid transparent;
    border-radius: 20px;
    overflow: hidden;
}

.threejs-container {
    width: 100%;
    height: 100vh;
}

.threejs-container canvas {
    width: 100% !important;
    height: 100% !important;
}

.contentTextTable {
    max-width: 120px;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
}

.title-channel {
    font-size: 19px;
    color: var(--user-color);
}

tr th {
    color: var(--user-color);
}

.color-title {
    color: gray;
    font-size: 20px;
    font-weight: bold;
}

.table-data {
    min-height: 20vh;
    overflow-y: scroll;
}

.name-avatar {
    display: flex;
    align-items: center;
    align-content: center;
}

.name-avatar img {
    width: 60px;
    height: 60px;
    object-fit: cover;
}

.DeleteMap .fa-trash:hover {
    transition: all 0.5s ease;
    color: red;
}

.DeleteMap .fa-trash-arrow-up:hover {
    transition: all 0.5s ease;
    color: green;
}

.DeleteMap {
    transition: all 0.5s ease;
    font-size: 22px;
}

.updateContent {
    transition: all 0.5s ease;
    font-size: 22px;
}

.updateContent .fa-pen:hover {
    transition: all 0.5s ease;
    color: #3366FF;
}

.view-detail-map {
    transition: all 0.5s ease;
    font-size: 22px;
}

.view-detail-map i:hover {
    transition: all 0.5s ease;
    color: var(--user-color);
}

.name-avatar img {
    border-radius: 6px;
}

.nameMember {
    margin-left: 10px;
}

#main {
    padding: 10px 20px;
}

#page {
    margin-right: auto;
}

table {
    font-size: 12px;
}

table img {
    max-width: 150px;
    height: auto;
    object-fit: cover;
}

.table-cell {
    font-weight: bold;
    vertical-align: middle;
}

table thead th,
table tbody th {
    vertical-align: middle;
    text-align: center;
}

.imgInTable {
    display: flex;
    justify-content: center;
}

.imgInTable img {
    border-radius: 4px;
}

table button {
    padding: 1px 3px;
    margin-right: 2px;
}

.colorImage {
    color: var(--blue-color);
}

.colorText {
    color: var(--brown-color)
}

.action {
    display: flex;
    align-items: center;
    justify-content: center;
}

.form-control {
    height: calc(1.5em + .5rem + 2px);
    padding: .25rem .5rem;
    font-size: .875rem;
    border-radius: 0.2rem;
    line-height: 1.5;
}

.row-multi-btn {
    display: flex;
}

#search-sort {
    line-height: 1.5;
}

@media screen and (min-width: 1201px) {
    table {
        max-width: 100%;
        vertical-align: middle;
    }

    td .fa-solid {
        font-size: 20px;
    }

    .contentTextTable {
        max-width: 150px;
    }
}

@media screen and (min-width: 993px) and (max-width: 1200px) {
    table {
        max-width: 100%;
        vertical-align: middle;
    }

    .name-avatar {
        min-width: 120px;
    }

    table {
        font-size: 11px;
    }

    .fa-solid {
        font-size: 15px;
    }

    .name-avatar img {
        min-width: 50px;
        min-height: 50px;
        max-width: 50px;
        max-height: 50px;
        object-fit: cover;
    }

    .table td,
    .table th {
        padding: 8px;
    }

    .form-control,
    .pagination {
        font-size: 12px !important;
    }

    .input-group-text {
        padding: 1px 9px;
    }

    #main {
        padding: 1% 1%;
        margin: 0;
    }

    .col-1,
    .col-2,
    .col-3 {
        padding-right: 8px;
    }

    table button {
        padding: 1px 2px;
    }

    table img {
        max-width: 110px;
    }

    .contentTextTable {
        max-width: 130px;
    }
}

@media screen and (min-width: 769px) and (max-width: 992px) {
    .title-channel {
        font-size: 15px;
    }

    table {
        max-width: 100%;
        vertical-align: middle;
    }

    .name-avatar {
        min-width: 140px;
    }

    table {
        font-size: 11px;
    }

    .fa-solid {
        font-size: 16px;
    }

    .name-avatar img {
        min-width: 50px;
        min-height: 50px;
        max-width: 50px;
        max-height: 50px;
        object-fit: cover;
    }

    .table td,
    .table th {
        padding: 8px;
    }

    .form-control,
    .pagination {
        font-size: 12px !important;
    }

    .input-group-text {
        padding: 0 6px;
    }

    #main {
        padding: 1% 1%;
        margin: 0;
    }

    #page {
        min-width: 65px;
    }

    .col-1,
    .col-2,
    .col-3 {
        padding-left: 0;
        padding-right: 3px;
    }

    .btn {
        padding: 1px 5px 0 5px;
    }

    table button {
        padding: 1px 2px;
    }

    table img {
        max-width: 100px;
    }

    .contentTextTable {
        max-width: 110px;
    }
}

@media screen and (min-width: 577px) and (max-width: 768px) {

    .title-channel,
    table {
        max-width: 100%;
        vertical-align: middle;
    }

    .name-avatar {
        min-width: 120px;
    }

    table {
        font-size: 11px;
    }

    .fa-solid {
        font-size: 13px;
    }

    .name-avatar img {
        min-width: 40px;
        min-height: 40px;
        max-width: 40px;
        max-height: 40px;
        object-fit: cover;
    }

    .table td,
    .table th {
        padding: 8px;
    }

    .form-control,
    .pagination {
        font-size: 12px !important;
    }

    #page {
        min-width: 45px;
    }

    .form-control {
        padding: 1px 1px;
    }

    #main {
        padding: 1% 1%;
        margin: 0;
    }

    .col-1,
    .col-2,
    .col-3 {
        padding-right: 5px;
    }

    .btn {
        padding: 1px 4px 0 4px;
    }

    .input-group-text {
        padding: 0 4px;
    }

    .input-group-prepend {
        font-size: 12px;

    }

    .mr-3 {
        margin-left: -1% !important;
        margin-right: 0px !important
    }

    table button {
        padding: 1px;
    }

    table img {
        max-width: 100px;
    }

    .contentTextTable {
        max-width: 110px;
    }
}

@media screen and (min-width: 425px) and (max-width: 576px) {

    .title-channel,
    table {
        max-width: 100%;
        vertical-align: middle;
    }

    .nameMember {
        margin-left: 8px;
    }

    .name-avatar {
        min-width: 100px;
    }

    table {
        font-size: 10px;
    }

    .fa-solid {
        font-size: 10px;
    }

    .name-avatar img {
        min-width: 40px;
        min-height: 40px;
        max-width: 40px;
        max-height: 40px;
        object-fit: cover;
    }

    .table td,
    .table th {
        padding: 5px;
    }

    .form-control,
    .pagination {
        font-size: 10px !important;
    }

    .form-control {
        padding: 1px 1px;
        height: 25px;
    }

    #page {
        min-width: 45px;
    }

    #main {
        padding: 1% 1%;
        margin: 0;
    }

    .col-1,
    .col-2,
    .col-3 {
        padding-right: 5px;
    }

    .btn {
        padding: 0px 4px;
    }

    .input-group-text {
        padding: 0 3px;
    }

    .input-group-prepend {
        font-size: 11px;
    }

    .mr-3 {
        margin-left: -2% !important;
        margin-right: 0px !important
    }

    table button {
        padding: 1px;
    }

    .mt-3 {
        margin-top: 0 !important;
    }

    table img {
        max-width: 80px;
    }

    .contentTextTable {
        max-width: 90px;
    }
}

@media screen and (min-width: 375px) and (max-width: 424px) {

    .title-channel,

    table {
        max-width: 100%;
        vertical-align: middle;
    }

    .nameMember {
        margin-left: 8px;
    }

    .name-avatar {
        min-width: 100px;
    }

    table {
        font-size: 9px;
    }

    .fa-solid {
        font-size: 10px;
    }

    .name-avatar img {
        min-width: 40px;
        min-height: 40px;
        max-width: 40px;
        max-height: 40px;
        object-fit: cover;
    }

    .table td,
    .table th {
        padding: 4px;
    }

    .form-control,
    .pagination {
        font-size: 9px !important;
    }

    .form-control {
        padding: 0.5px 0;
        height: 25px;
    }

    #page {
        min-width: 40px;
    }

    #main {
        padding: 1% 1%;
        margin: 0;
    }

    .col-1,
    .col-2,
    .col-3 {
        padding-right: 0;
    }

    .btn {
        padding: 0px 4px;
    }

    .input-group-text {
        padding: 0 2px;
    }

    .input-group-prepend {
        font-size: 10px;

    }

    #main .ml-2 {
        margin-left: 3px !important;
    }

    .mr-3 {
        margin-left: 0px !important;
        margin-right: 0px !important;
    }

    table button {
        padding: 0.7px;
    }

    .mt-3 {
        margin-top: 0 !important;
    }

    table img {
        max-width: 70px;
    }

    .contentTextTable {
        max-width: 80px;
    }
}
</style>
