<template>
    <div id="main">
        <div>
            <h3 class="title-channel"><i class="fa-regular fa-image"></i> Manage Map of System </h3>
        </div>
        <div class="ml-2 mt-2">
            <div class="mt-3">
                <div class="row m-0 pb-2 d-flex justify-content-end gap-2" id="search-sort">
                    <div class="pl-0" id="page">
                        <select content="Pagination" v-tippy class="form-control " v-model="big_search.perPage">
                            <option value="5">5</option>
                            <option value="10">10</option>
                            <option value="15">15</option>
                            <option value="20">20</option>
                        </select>
                    </div>
                    <div class="pl-0 mr-3">
                        <select content="Sort by" v-tippy class="form-control " v-model="big_search.typesort">
                            <option value="new">New</option>
                            <option value="name">Name</option>
                        </select>
                    </div>
                    <div class="pl-0 mr-3">
                        <select content="In direction" v-tippy class="form-control " v-model="big_search.sortlatest">
                            <option value="false">Ascending</option>
                            <option value="true">Decrease</option>
                        </select>
                    </div>
                    <div class="pl-0 ">
                        <div content="Search information content" v-tippy class="input-group">
                            <div class="input-group-prepend">
                                <div class="input-group-text"><i class="fa-solid fa-magnifying-glass"></i></div>
                            </div>
                            <input v-model="search" type="text" class="form-control " id="inlineFormInputGroup"
                                placeholder="Search...">
                        </div>
                    </div>
                    <div class="pr-0 " v-if="selectedMaps.length > 0">
                        <div class="input-group">
                            <button @click="changeDeleteManyMap()" content="Delete Many Map" v-tippy
                                data-toggle="modal" data-target="#modal-delete-many-map" type="button"
                                class="btn btn-outline-danger mr-1 form-control"><i
                                    class="fa-solid fa-trash"></i></button>
                        </div>
                    </div>
                    <div class="mr-3">
                        <div class="input-group ">
                            <button content="Add Map" v-tippy data-toggle="modal" data-target="#modal-add-map"
                                type="button" class="btn btn-success form-control"><i
                                    class="fa-solid fa-plus"></i></button>
                        </div>
                    </div>
                </div>
                <div v-if="isLoading">
                    <TableLoading :cols="6" :rows="9"></TableLoading>
                </div>
                <div v-if="!isLoading" class="table-data">
                    <table class="table table-bordered">
                        <thead>
                            <tr>
                                <th scope="col"><input ref="selectAllCheckbox" @change="selectAll()" type="checkbox" class=""></th>
                                <th scope="col">#</th>
                                <th scope="col"><i class="fa-solid fa-map-location-dot"></i> Adress</th>
                                <th scope="col"><i class="fa-solid fa-panorama"></i> Pandora 360</th>
                                <th scope="col"><i class="fa-solid fa-map-pin"></i> Longitude & Latitude</th>
                                <th scope="col" class="text-center"><i class="fa-solid fa-calendar-day"></i> Created at</th>
                                <th scope="col" class="text-center"><i class="fa-solid fa-calendar-check"></i> Updated at</th>
                                <th scope="col" class="text-center"><i class="fa-solid fa-user-pen"></i> Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(map, index) in maps" :key="index">
                                <th class="table-cell" scope="row"><input :checked="isSelected(map.id)"
                                        type="checkbox" class="" @change="handleSelect(map.id)"></th>
                                <th class="table-cell" scope="row">#{{ (big_search.page - 1) * big_search.perPage +
                                index +
                                1 }}
                                </th>
                                <td class="table-cell text-center">{{ map.address }}</td>
                                <td class="table-cell text-center"><img :src="map.imagefull ? map.imagefull : require('@/assets/pandora360.png')" alt=""></td>
                                <td class="table-cell text-center">[ {{ map.coordinates[0] }}, {{ map.coordinates[1] }} ]</td>
                                <td class="table-cell text-center">{{ formatDate(map.created_at) }}</td>
                                <td class="table-cell text-center">{{ formatDate(map.updated_at) }}</td>
                                <td class="table-cell text-center">
                                    <div class="action">
                                        <button data-toggle="modal" data-target="#modal-view-detail-map"
                                            v-tippy="{ content: 'View Detail' }" class="view-detail-map text-success"
                                            @click="selectMap(map,'detail')">
                                            <i class="fa-solid fa-eye"></i>
                                        </button>
                                        <button data-toggle="modal" data-target="#modal-update-map"
                                            v-tippy="{ content: 'Update' }" class="updateContent text-primary"
                                            @click="selectMap(map,'update')">
                                            <i class="fa-solid fa-pen"></i>
                                        </button>
                                        <button data-toggle="modal" data-target="#modal-delete-map"
                                            v-tippy="{ content: 'Delete' }"
                                            class="DeleteMap text-danger" @click="selectMap(map,'delete')">
                                            <i class="fa-solid fa-trash"></i>
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div id="divpaginate" class="mt-2">
                    <paginate v-if="paginateVisible" :page-count="last_page" :page-range="3" :margin-pages="2"
                        :click-handler="clickCallback" :initial-page="big_search.page" :prev-text="'Prev'"
                        :next-text="'Next'" :container-class="'pagination'" :page-class="'page-item'">
                    </paginate>
                </div>
                <MapAdd></MapAdd>
                <MapUpdate :mapSelected="mapSelected"></MapUpdate>
                <DeleteMap :mapSelected="mapSelected"></DeleteMap>
                <DeleteManyMap :selectedMaps="selectedMaps"></DeleteManyMap>
            </div>
        </div>
    </div>
</template>

<script>
import useEventBus from '@/composables/useEventBus'
import UserRequest from '@/restful/UserRequest';
const { emitEvent, onEvent } = useEventBus();
import Paginate from 'vuejs-paginate-next';
import config from '@/config';
import TableLoading from '@/components/common/TableLoading'
import _ from 'lodash';
import MapAdd from '@/components/user/geo/MapAdd.vue'
import MapUpdate from '@/components/user/geo/MapUpdate.vue'
import DeleteMap from '@/components/user/geo/DeleteMap.vue'
import DeleteManyMap from '@/components/user/geo/DeleteManyMap.vue'

export default {
    name: "MapManage",
    components: {
        paginate: Paginate,
        TableLoading,
        DeleteMap,
        DeleteManyMap,
        MapAdd,
        MapUpdate,
    },

    data() {
        return {
            config: config,
            total: 0,
            last_page: 1,
            paginateVisible: true,
            search: '',
            big_search: {
                perPage: 5,
                page: 1,
                typesort: 'new',
                sortlatest: 'true',
            },
            query: '',
            maps: [],
            mapSelected : {
                id : null,
                address : null,
                coordinates : null,
                imagefull: null,
                pz : null,
                nz : null,
                px : null,
                nx : null,
                py : null,
                ny : null,
            },
            selectedMaps: [],
            isLoading: false,
            isDeleteChangeMany: 0,
        }
    },

    setup() {
        document.title = "Manages Maps | PBL7"
    },

    mounted() {
        emitEvent('eventTitleHeader', 'Channel Manages Maps');
        const queryString = window.location.search;
        const searchParams = new URLSearchParams(queryString);
        this.search = searchParams.get('search') || '';
        this.big_search = {
            perPage: parseInt(searchParams.get('paginate')) || 5,
            page: searchParams.get('page') || 1,
            typesort: searchParams.get('typesort') || 'new',
            sortlatest: searchParams.get('sortlatest') || 'true',
        }
        this.getMaps();
        onEvent('eventRegetMaps', () => {
            this.getMaps();
        });
    },

    methods: {
        reRenderPaginate: function () {
            if (this.big_search.page > this.last_page) this.big_search.page = this.last_page;
            this.paginateVisible = false;
            this.$nextTick(() => { this.paginateVisible = true; });
        },
        getMaps: async function () {
            this.selectedMaps = [];
            this.isLoading = true;
            this.query = '?search=' + this.search + '&typesort=' + this.big_search.typesort + '&sortlatest=' + this.big_search.sortlatest
                 + '&paginate=' + this.big_search.perPage + '&page=' + this.big_search.page;
            window.history.pushState({}, null, this.query);

            try {
                const { data, messages } = await UserRequest.get('map/all' + this.query)
                this.maps = data.data
                this.total = data.total;
                this.last_page = data.last_page;
                this.isLoading = false;
                emitEvent('eventSuccess', messages[0]);
            }
            catch (error) {
                if (error.messages) emitEvent('eventError', error.messages[0]);
                this.isLoading = false;
            }
            this.reRenderPaginate();
        },
        truncatedTitle(title) {
            const maxLength = 150;
            if (title.length > maxLength) return title.slice(0, maxLength) + '...';
            else return title;
        },

        formatDate: function (date) {
            const formattedDate = new Date(date);

            const day = formattedDate.getDate();
            const month = formattedDate.getMonth() + 1;
            const year = formattedDate.getFullYear();

            const formattedDateString = `${day}/${month}/${year}`;

            return formattedDateString;
        },

        clickCallback: function (pageNum) {
            this.big_search.page = pageNum;
        },

        selectMap: function (mapSelected, status) {
            if(status == 'detail') {
                this.$router.push({ name: 'ViewDetailMap', params: { id: mapSelected.id } });
            }
            else if(status == 'delete') {
                emitEvent('selectSimpleMapDelete', mapSelected);
            }
            else{
                emitEvent('selectSimpleMap', mapSelected);
            }
            this.mapSelected = mapSelected;
        },

        isSelected(mapId) {
            return this.selectedMaps.includes(mapId);
        },

        handleSelect: function (mapId) {
            const index = this.selectedMaps.indexOf(mapId);
            if (index === -1) this.selectedMaps.push(mapId);
            else this.selectedMaps.splice(index, 1);
        },

        selectAll: function () {
            const checkbox = this.$refs.selectAllCheckbox;
            if (checkbox.checked) this.selectedMaps = this.maps.map(map => map.id);
            else this.selectedMaps = [];
        },

        changeDeleteManyMap: function () {
            emitEvent('selectManyMap', this.maps);
        }

    },
    watch: {
        big_search: {
            handler: function () {
                this.getMaps();
            },
            deep: true
        },
        search: _.debounce(function () {
            this.getMaps();
        }, 500),
    }
}
</script>

<style scoped>
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

    .contentTextTable{
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

    .input-group-text{
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

    .contentTextTable{
        max-width: 130px;
    }
}

@media screen and (min-width: 769px) and (max-width: 992px) {
    .title-channel {
        font-size: 15px;
    }

    .color-title {
        font-size: 14px;
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

    .contentTextTable{
        max-width: 110px;
    }
}

@media screen and (min-width: 577px) and (max-width: 768px) {

    .title-channel,
    .color-title {
        font-size: 13px;
    }

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

    .contentTextTable{
        max-width: 110px;
    }
}

@media screen and (min-width: 425px) and (max-width: 576px) {

    .title-channel,
    .color-title {
        font-size: 12px;
    }

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

    .mt-3{
        margin-top: 0 !important;
    }

    table img {
        max-width: 80px;
    }

    .contentTextTable{
        max-width: 90px;
    }
}

@media screen and (min-width: 375px) and (max-width: 424px) {

    .title-channel,
    .color-title {
        font-size: 11px;
    }

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

    .mt-3{
        margin-top: 0 !important;
    }

    table img {
        max-width: 70px;
    }

    .contentTextTable{
        max-width: 80px;
    }
}
</style>
