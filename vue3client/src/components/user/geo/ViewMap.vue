<template>
    <div @mousedown="handleClickOutside" @touchstart="handleClickOutside">
        <div>
            <div v-if="!show"  class="left-bar">
                <div class="list-feature">
                    <button type="button" class="btn btn-outline-primary mr-1" @click="toggleMap">Show Map</button>
                    <button type="button" class="btn btn-outline-primary mr-1"><router-link :to="{ name: 'MapManage' }"><i class="fa-solid fa-sliders"></i> Map Manage</router-link></button>
                    <div class="search-map">
                        <Multiselect v-model="mapSelected" placeholder="Search for locations" 
                        :searchable="true" :create-option="true" :options="arrAddress" />
                        <!-- trong Multiselect sẽ có label và value (xem ở chỗ gán giá trị cho arrAddress) , 
                        label là hiển thị tên option , còn value là giá trị cho option đó , value có thể là arr, object, number,..  -->
                    </div>
                        <!-- bỏ :close-on-select="false" và mode="tags" sẽ trở thành single select -->
                </div>
            </div>
            <div class="bg-content-map" v-if="show">
                    <div class="content-map" ref="contentMap" >
                    <SelectMap :maps="maps"></SelectMap>
                </div>
                </div>
            <LoadMap></LoadMap>
        </div>
    </div>
</template>

<script>
import LoadMap from '@/components/user/geo/LoadMap.vue'
import SelectMap from '@/components/user/geo/SelectMap.vue'
import UserRequest from '@/restful/UserRequest';
import useEventBus from '@/composables/useEventBus'
const { emitEvent, onEvent } = useEventBus();
import Multiselect from '@vueform/multiselect'

export default {
    name: "ViewMap",
    components: {
        LoadMap,
        SelectMap,
        Multiselect
    },
    data() {
        return {
            mapSelected: null,
            show: false,
            arrAddress: [],
        }
    },
    mounted() {
        onEvent('selectPoint', (map) => {
            console.log(map);
            this.show = false;
        });
        this.getMaps();
    },
    methods: {
        toggleMap() {
            this.show = !this.show;
        },
        handleClickOutside(event) {
            const contentMap = this.$refs.contentMap;
            if (contentMap && !contentMap.contains(event.target)) {
                this.show = false;
            }
        },
        getMaps: async function () {
            try {
                const { data, messages } = await UserRequest.get('map/all')
                this.maps = data
                console.log(messages);
                // emitEvent('eventSuccess', messages[0]);
                data.forEach((map) => {
                    var option = { value: map, label: map.address }
                    this.arrAddress.push(option);
                });
            }
            catch (error) {
                if (error.messages) emitEvent('eventError', error.messages[0]);
            }
        }
    },
    watch: {
        mapSelected: function (mapSelected) {
            console.log(mapSelected);
            if(mapSelected) {
                emitEvent('selectPoint', mapSelected);
            }
        },
        show(newVal) {
            if (newVal) {
                document.addEventListener('mousedown', this.handleClickOutside);
                document.addEventListener('touchstart', this.handleClickOutside);
            } else {
                document.removeEventListener('mousedown', this.handleClickOutside);
                document.removeEventListener('touchstart', this.handleClickOutside);
            }
        }
    }
}
</script>

<style scoped>
.search-map {
    width: 300px;
}
button a:hover {
    color: white;
    text-decoration: none;
}
.list-feature {
    width: fit-content;
    display: flex;
}
.left-bar {
    background-color: white;
    padding: 10px;
    border-bottom-right-radius: 10px;
    position: fixed;
    top: 0px;
    left: 0px;
    width: fit-content;
    height: fit-content;
}
.bg-content-map {
    position: fixed;
    z-index: 2;
    width: 100vw;
    height: 100vh;
    background-color: rgba(151, 151, 151, 0.579);
}
.content-map {
    position: absolute;
    width: 90vw;
    top: 10vh;
    left: 5vw;
    padding: 6px;
    border-radius: 10px;
    background-color: white;
    height: fit-content;
    margin: auto;
    overflow: hidden;
}
</style>
