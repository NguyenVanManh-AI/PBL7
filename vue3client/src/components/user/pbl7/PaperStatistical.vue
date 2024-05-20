<template>
    <div id="main">
        <div>
            <h3 class="title-channel"><i class="fa-solid fa-square-poll-vertical"></i> Statistics of scientific articles
            </h3>
        </div>
        <div class="mt-2 row">
            <div class="col-8">
                <div class="form-group mb-0 pb-">
                    <label class="font-weight-bold" for="exampleFormControlSelect1"><i class="fa-solid fa-arrow-trend-up"></i> Trend of top 10
                        keywords Over 10 Years</label>
                    <select v-model="yearTop10Trend" @change="getImageTop10Trend" class="form-control col-10"
                        id="exampleFormControlSelect1">
                        <option v-for="year in years" :key="year">{{ year }}</option>
                    </select>
                </div>
                <flower-spinner v-if="!urlTop10Trend" class="loading-component"
                                        :animation-duration="2000" :size="50" color="#06C755" />
                <img v-if="urlTop10Trend" :src="computedUrlTop10Trend" alt="Trend Image">
            </div>
        </div>
        <div class="mt-4 row">
            <div class="col-4">
                <label class="font-weight-bold" for="exampleFormControlSelect1"><i class="fa-solid fa-chart-column"></i> Keyword table and related keyword list</label>
                <select v-model="big_search.year" @change="getKeyWords" class="form-control mb-1" 
                        id="exampleFormControlSelect1">
                        <option v-for="year in years" :key="year">{{ year }}</option>
                    </select>
                <div v-if="isLoading">
                    <TableLoading :cols="6" :rows="9"></TableLoading>
                </div>
                <div v-if="!isLoading" class="table-data">
                    <table class="table table-bordered">
                        <thead>
                            <tr>
                                <th scope="col">#</th>
                                <th scope="col"><i class="fa-solid fa-key"></i> Key Word </th>
                                <th scope="col"><i class="fa-solid fa-chart-simple"></i> Frequency</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(keyword, index) in keywords" :key="index">
                                <th class="table-cell" scope="row">#{{ (big_search.page - 1) * big_search.perPage + index + 1 }}</th>
                                <td @click="getImageKeywordRelated(keyword.Word)" class="name-keyword table-cell text-center"><span>{{ keyword.Word }}</span></td>
                                <td class="table-cell text-center">{{ keyword.Frequency }}</td>
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
            </div>
            <div class="col-8" >
                <p class="text-center font-weight-bold" for="exampleFormControlSelect1"><i class="fa-solid fa-circle"></i> Visualize wordcloud for keywords</p>
                <div class="row img-wordcloud">
                    <flower-spinner v-if="!urlWordcloud" class="loading-component"
                                        :animation-duration="2000" :size="50" color="#06C755" />
                    <img v-if="urlWordcloud" :src="computedUrlWordcloud" alt="Trend Image">
                </div>
                <p v-if="urlKeyWordRelated" class="text-center p-label-keyword font-weight-bold" for="exampleFormControlSelect1"><i class="fa-solid fa-chart-line"></i> Line graph for related keywords</p>
                <div class="row img-keyword">
                    <flower-spinner v-if="!urlKeyWordRelated" class="loading-component"
                                    :animation-duration="2000" :size="50" color="#06C755" />
                    <img v-if="urlKeyWordRelated" :src="computedUrlKeyWordRelated" alt="Trend Image">
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import ModelRequest from '@/restful/ModelRequest';
import useEventBus from '@/composables/useEventBus';
import config from '@/config';
import Paginate from 'vuejs-paginate-next';
import TableLoading from '@/components/common/TableLoading'
import { FlowerSpinner } from 'epic-spinners';

const { emitEvent } = useEventBus();

export default {
    name: "PaperSearch",
    components: {
        paginate: Paginate,
        FlowerSpinner,
        TableLoading,
    },
    data() {
        return {
            config: config,
            years: [],
            // 
            yearTop10Trend: 2023,
            urlTop10Trend: '',
            // 
            total: 0,
            last_page: 1,
            paginateVisible: true,
            big_search: {
                page: 1,
                year: 2023,
                perPage: 20,
            },
            query: '',
            keywords: [],
            urlWordcloud: '', 
            urlKeyWordRelated: '', 
            isLoading: false,
        }
    },
    created() {
        const startYear = 1987;
        const endYear = 2023;
        for (let year = startYear; year <= endYear; year++) this.years.unshift(year);
        this.getImageTop10Trend(); // Load the initial trend
    },
    mounted() {
        emitEvent('eventTitleHeader', 'Statistics of scientific articles');
        document.title = "Statistics of scientific articles | PBL7";
        const queryString = window.location.search;
        const searchParams = new URLSearchParams(queryString);
        this.big_search = {
            perPage: parseInt(searchParams.get('paginate')) || 20,
            page: searchParams.get('page') || 1,
            year: searchParams.get('year') || 2023,
        }
        this.getKeyWords();
    },
    computed: {
        computedUrlTop10Trend() {
            // Thêm một chuỗi truy vấn ngẫu nhiên để đảm bảo ảnh được tải lại
            return this.urlTop10Trend + '?t=' + new Date().getTime();
        },
        computedUrlKeyWordRelated() {
            return this.urlKeyWordRelated + '?t=' + new Date().getTime();
        },
        computedUrlWordcloud() {
            return this.urlWordcloud + '?t=' + new Date().getTime();
        },
    },
    methods: {
        async getImageTop10Trend() {
            try {
                this.urlTop10Trend = '';
                const { url } = await ModelRequest.get('trend_10_year?year=' + this.yearTop10Trend, false);
                // bằng cách này v-if hoặc động và xóa img khỏi DOM rồi tạo lại để load ra ảnh mới 
                this.$nextTick(() => {
                    this.urlTop10Trend = url;
                });
                emitEvent('eventSuccess', 'Get Trend of top 10 keywords Over 10 Years success!');
            } catch {
                emitEvent('eventError', 'Get Trend of top 10 keywords Over 10 Years fail !');
            }
        },
        reRenderPaginate: function () {
            if (this.big_search.page > this.last_page) this.big_search.page = this.last_page;
            this.paginateVisible = false;
            this.$nextTick(() => { this.paginateVisible = true; });
        },
        getKeyWords: async function () {
            this.isLoading = true;
            this.query = '?year=' + this.big_search.year + '&page=' + this.big_search.page;
            window.history.pushState({}, null, this.query);
            try {
                this.urlWordcloud = '';
                const { results, count } = await ModelRequest.get('trend_year' + this.query, false)
                console.log(results.table);
                this.keywords = results.table;
                this.urlWordcloud = results.url;
                this.total = count;
                this.last_page = Math.ceil(count / this.big_search.perPage); // trang cuối 
                this.isLoading = false;
                emitEvent('eventSuccess', 'Get keyword table success !');
            }
            catch(error) {
                console.log(error);
                emitEvent('eventError', 'Get keyword table fail !');
            }
            this.reRenderPaginate();
        },
        async getImageKeywordRelated(keyword) {
            try {
                this.urlKeyWordRelated = '';
                const { url } = await ModelRequest.get('trend_10_keywords?year=' + this.big_search.year+'&keyword=' + keyword, false);
                this.$nextTick(() => {
                    this.urlKeyWordRelated = url;
                });
                emitEvent('eventSuccess', 'Get KeyWord Related success !');
            } catch {
                emitEvent('eventError', 'Get KeyWord Related fail !');
            }
        },
        clickCallback: function (pageNum) {
            this.big_search.page = pageNum;
        },
    },
    watch: {
        big_search: {
            handler: function () {
                this.getKeyWords();
            },
            deep: true
        },
    }
}
</script>

<style scoped>
.loading-component {
    margin-top: 20px !important;
}

.img-wordcloud {
    overflow: hidden;
}

.img-keyword {
    overflow: hidden;
}

.img-keyword img {
    scale: 1.1;
    margin-top: -40px;
    margin-left: -20px;
}

.img-wordcloud img {
    scale: 1.2;
    margin-top: -130px;
    margin-left: -20px;
}

.name-keyword span {
    color: #007BFF;
}
.name-keyword:hover {
    color: #007BFF;
    cursor: pointer;
    text-decoration: underline;
}

.title-channel {
    font-size: 19px;
    color: var(--user-color);
}

tr th {
    color: var(--user-color);
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

table button {
    padding: 1px 3px;
    margin-right: 2px;
}

.form-control {
    height: calc(1.5em + .5rem + 2px);
    padding: .25rem .5rem;
    font-size: .875rem;
    border-radius: 0.2rem;
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
}

@media screen and (min-width: 993px) and (max-width: 1200px) {
    table {
        max-width: 100%;
        vertical-align: middle;
    }

    table {
        font-size: 11px;
    }

    .fa-solid {
        font-size: 15px;
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

}

@media screen and (min-width: 769px) and (max-width: 992px) {
    .title-channel {
        font-size: 15px;
    }

    table {
        max-width: 100%;
        vertical-align: middle;
    }

    table {
        font-size: 11px;
    }

    .fa-solid {
        font-size: 16px;
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

}

@media screen and (min-width: 577px) and (max-width: 768px) {

    .title-channel,
    table {
        max-width: 100%;
        vertical-align: middle;
    }

    table {
        font-size: 11px;
    }

    .fa-solid {
        font-size: 13px;
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

}

@media screen and (min-width: 425px) and (max-width: 576px) {

    .title-channel,
    table {
        max-width: 100%;
        vertical-align: middle;
    }

    table {
        font-size: 10px;
    }

    .fa-solid {
        font-size: 10px;
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

}

@media screen and (min-width: 375px) and (max-width: 424px) {

    .title-channel,

    table {
        max-width: 100%;
        vertical-align: middle;
    }

    table {
        font-size: 9px;
    }

    .fa-solid {
        font-size: 10px;
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

}
</style>