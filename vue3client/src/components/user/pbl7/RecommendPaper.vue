<template>
    <div id="recommend_paper">
        <button id="open_recommend" type="button" class="btn btn-outline-warning" data-toggle="modal"
            data-target="#exampleModal"><span><i class="fa-solid fa-lightbulb"></i></span></button>
        <div class="modal fade" id="exampleModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel"
            aria-hidden="true">
            <div class="modal-dialog" role="document">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title text-warning" id="exampleModalLabel"><strong><i class="fa-solid fa-star"></i>
                                Recommend Papers - These papers may be suitable for you !</strong></h5>
                        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                            <span aria-hidden="true">&times;</span>
                        </button>
                    </div>
                    <div class="modal-body">
                        <div id="carouselExampleControls" class="carousel slide" data-ride="carousel">
                            <div class="carousel-inner">
                                <div class="carousel-item" v-for="(paper, index) in papers"
                                    :class="{ active: index === 0 }" :key="paper.ID">
                                    <div class="innter-item">
                                        <div class="text-center text-success col-8 mx-auto title-paper-recommned">
                                            <h2><strong><i class="fa-solid fa-file-circle-check"></i> {{ paper.Title }}</strong></h2>
                                        </div>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Authors</span> : {{ paper['Authors'] }}</p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Year</span> : {{ paper['Year'] }}</p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Status</span> : {{ paper['Status'] }}</p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Book Title</span> : {{ paper['Book Title'] }}</p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Editors</span> : {{ paper['Editors'] }}</p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Publishers</span> : {{ paper['Publishers'] }}</p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Main Url</span> : <a target="_blank" :href="paper['Main Url']">{{ paper['Main Url'] }}</a></p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Paper Url</span> : <a target="_blank" :href="paper['Paper Url']">{{ paper['Paper Url'] }}</a></p>
                                        <p class="infor-paper"><span><i class="fa-regular fa-circle-check"></i> Supplemental Url</span> : <a target="_blank" :href="paper['Supplemental Url']">{{ paper['Supplemental Url'] }}</a></p>
                                        <div class="card mt-2">
                                            <div class="card-header" :id="'accordion' + paper.ID">
                                                <h5 class="mb-0">
                                                    <button class="btn btn-link" data-toggle="collapse"
                                                        :data-target="'#card' + paper.ID" aria-expanded="true"
                                                        aria-controls="collapseOne">
                                                        Show More
                                                    </button>
                                                </h5>
                                            </div>
                                            <div :id="'card' + paper.ID" class="collapse show"
                                                aria-labelledby="headingOne" :data-parent="'#accordion' + paper.ID">
                                                <div class="card-body">
                                                    <p class="p-abstract" :ref="'abstract' + nth"><span
                                                            style="font-weight:bold"><i
                                                                class="fa-solid fa-quote-left"></i> Abstract </span>: {{
                                                        paper.Abstract }}</p>
                                                    <p class="p-keywords" :ref="'keywords' + nth"><span
                                                            style="font-weight:bold"><i class="fa-solid fa-key"></i>
                                                            Keywords </span>: {{ paper.Keywords }}</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <a class="carousel-control-prev" href="#carouselExampleControls" role="button"
                                data-slide="prev">
                                <span class="btn-slide"><i class="fa-solid fa-angles-left"></i></span>
                            </a>
                            <a class="carousel-control-next" href="#carouselExampleControls" role="button"
                                data-slide="next">
                                <span class="btn-slide"><i class="fa-solid fa-angles-right"></i></span>
                            </a>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary btn-sm" data-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
import config from '@/config';
// import useEventBus from '@/composables/useEventBus';
import ModelRequest from '@/restful/ModelRequest';
import UserRequest from '@/restful/UserRequest';
// const { emitEvent } = useEventBus();
export default {
    name: "RecommendPaper",
    data() {
        return {
            config: config,
            papers: null,
        }
    },
    components: {

    },
    mounted() {
        this.getKeyWork();
    },
    methods: {
        async getKeyWork() {
            try {
                var { data, messages } = await UserRequest.get('tracking/get');
                var keyword = data.name ?? 'new_user';
                console.log(keyword);
                this.getPaper(keyword);
                console.log(data, messages);
            } catch (error) {
                console.log('error', error)
            }
        },
        async getPaper(keyword) {
            try {
                var submit_data = { "keyword": keyword }
                console.log(submit_data);
                var { results } = await ModelRequest.post('recommender', submit_data);
                this.papers = results;
                console.log(this.papers);
                console.log(results);
            } catch (error) {
                console.log('error', error)
            }
        },
    }
}
</script>
<style scoped>
#recommend_paper {
    z-index: 999999;
}

.modal-dialog {
    max-width: 1200px;
}

#open_recommend {
    background-color: white;
    border-radius: 100px;
    position: fixed;
    right: 2%;
    top: 12%;
}

.carousel-control-next,
.carousel-control-prev {
    width: fit-content;
}

.carousel-control-prev span,
.carousel-control-next span {
    /* background-color: #28A644; */
    border: 1px solid #28A644;
    color: #027102;
    font-size: 20px;
    border-radius: 300px;
    padding: 6px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: auto;
    opacity: 0;
    transition: all 1s ease;
}

#carouselExampleControls:hover .btn-slide {
    opacity: 1;
    transition: all 1s ease;
}

#carouselExampleControls {
    position: relative;
}

#carouselExampleControls:before {
    content: ' ';
    display: block;
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    opacity: 0.2;
    background-image: url('~@/assets/neural_network_logo.png');
    background-repeat: no-repeat;
    background-position: 50% 0;
    background-size: cover;
}

.carousel-item {
    padding: 0px 42px;
    background-color: white;
}

.innter-item {
    height: 500px;
    overflow: hidden;
    overflow-y: scroll;
}

::-webkit-scrollbar-track {
    background: transparent;
}

.innter-item::-webkit-scrollbar-thumb {
    display: none;
}

.title-paper-recommned {
    font-size: 20px;
}

.infor-paper span {
    font-weight: bold;
}
.infor-paper  a {
  color: #0069D9 !important;
}
.card-header {
    padding: 0 !important;
}
</style>