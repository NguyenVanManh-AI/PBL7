<template>
    <div style="height: 80vh; width: 100%">
        <l-map ref="map" v-model:zoom="zoom" :center="center">
            <l-tile-layer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" layer-type="base"
                name="OpenStreetMap"></l-tile-layer>

            <!-- Hiển thị các marker từ danh sách điểm -->
            <l-marker v-for="(map, index) in maps" :key="index" :lat-lng="map.coordinates" @click="showPopup(map)">
                <l-popup>{{ map.address }}</l-popup>
                <l-tooltip>{{ map.address }}</l-tooltip>
            </l-marker>
        </l-map>
    </div>
</template>

<script>
import "leaflet/dist/leaflet.css";
import { LMap, LTileLayer, LMarker, LPopup, LTooltip } from "@vue-leaflet/vue-leaflet";
import useEventBus from '@/composables/useEventBus'
const { emitEvent } = useEventBus();

export default {
    name: "SelectMap",
    props : {
        maps: Array,
    },
    components: {
        LMap,
        LTileLayer,
        LMarker,
        LPopup,
        LTooltip,
    },
    data() {
        return {
            zoom: 12,
            selectedMap: null,
            center:[16.074954, 108.153419],
        };
    },
    mounted() {

    },
    methods: {
        showPopup(map) {
            this.selectedMap = map;
            console.log(map);
            emitEvent('selectPoint', map);
        },
    },
};
</script>
<style scoped>

</style>