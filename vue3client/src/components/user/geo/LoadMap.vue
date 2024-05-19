<template>
  <div><div ref="threejsContainer" class="threejs-container"></div></div>
</template>

<script>
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import useEventBus from '@/composables/useEventBus'
const { onEvent } = useEventBus();

export default {
  name: 'LoadMap',
  data() {
    return {
      renderer: null,
      scene: null,
      camera: null,
      cubeCamera: null,
      lightProbe: null,
      urls : [
        'https://linebotpro.s3.us-east-2.amazonaws.com/linebot/1715788198blob',
        'https://linebotpro.s3.us-east-2.amazonaws.com/linebot/1715788200blob',
        'https://linebotpro.s3.us-east-2.amazonaws.com/linebot/1715788201blob',
        'https://linebotpro.s3.us-east-2.amazonaws.com/linebot/1715788203blob',
        'https://linebotpro.s3.us-east-2.amazonaws.com/linebot/1715788206blob',
        'https://linebotpro.s3.us-east-2.amazonaws.com/linebot/1715788207blob'
      ],
    };
  },
  mounted() {
    this.init();
    window.addEventListener('resize', this.onWindowResize);
    onEvent('selectPoint', (map) => {
      console.log(map);
      this.urls = [
        map.px, map.nx,
        map.py, map.ny,
        map.pz, map.nz
      ];
      this.loadMap();
    });
  },
  beforeUnmount() {
    window.removeEventListener('resize', this.onWindowResize);
    this.dispose();
  },
  methods: {
    loadMap() {
      this.dispose();
      this.init();
    },
    init() {
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
      new THREE.CubeTextureLoader().load(this.urls, (cubeTexture) => {
        this.scene.background = cubeTexture;
        this.cubeCamera.update(this.renderer, this.scene);
        this.renderScene();
      });
    },
    dispose() {
      if (this.renderer) {
        this.renderer.dispose();
        this.renderer.domElement.remove();
      }
      if (this.scene) {
        this.scene.traverse((object) => {
          if (!object.isMesh) return;
          object.geometry.dispose();

          if (object.material.isMaterial) {
            this.cleanMaterial(object.material);
          } else {
            for (const material of object.material) this.cleanMaterial(material);
          }
        });
      }
    },
    cleanMaterial(material) {
      material.dispose();

      for (const key of Object.keys(material)) {
        const value = material[key];
        if (value && typeof value === 'object' && 'minFilter' in value) {
          value.dispose();
        }
      }
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
  },
};
</script>
<style scoped>
.threejs-container {
    width: 100%;
    height: 100vh;
}
.threejs-container canvas {
  width: 100% !important;
  height: 100% !important;
}
</style>