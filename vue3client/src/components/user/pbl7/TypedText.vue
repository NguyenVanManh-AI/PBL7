<template>
  <div class="container-question" v-if="content.type == 'question'">
    <p data-toggle="modal" :data-target="'#openModalKeywords' + content.idModal" class="additional-content"><i
        class="fa-solid fa-circle-question"></i> {{ content.contentvalue }}</p>
  </div>
  <div class="search-by" v-if="content.type == 'search_by'">
    <p class="search-line"></p>
    <p class="search-by-content">{{ handleCaseNotResult(content.valueSearchBy) }}</p>
  </div>
  <div v-if="content.type == 'result'" class="modal fade" :id="'openModalKeywords' + content.contentvalue.idModal"
    tabindex="-1" role="dialog" aria-labelledby="openModalKeywordsLabel" aria-hidden="true">
    <div class="modal-dialog" role="document">
      <div class="modal-content">
        <div class="modal-body">
          <div v-if="this.content.contentvalue.keywords.length > 0">
            <ul class="container-keywords">
              <span class="name_keyword_span" v-for="(value, key) in content.contentvalue.keywords" :key="key">
                {{ value }}
              </span>
            </ul>
          </div>
          <div v-else>
            <span class="name_keyword_span">{{ content.contentvalue.question }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div v-if="content.type == 'result'" :id="typedId" class="paper-details">
    <p :content="handleCaseNotResult(content.contentvalue.search_by)" v-tippy class="p-title"><span
        style='font-weight:bold'><i class="fa-solid fa-bookmark"></i></span> <span :ref="'title' + nth"></span></p>
    <p class="p-authors" :ref="'authors' + nth"></p>
    <p class="p-year" :ref="'year' + nth"></p>
    <p class="p-volume" :ref="'volume' + nth"></p>
    <p class="p-pages" :ref="'pages' + nth"></p>
    <p class="p-status" :ref="'status' + nth"></p>
    <p class="p-bookTitle" :ref="'bookTitle' + nth"></p>
    <p class="p-editors" :ref="'editors' + nth"></p>
    <p class="p-publishers" :ref="'publishers' + nth"></p>
    <p class="p-mainUrl" :ref="'mainUrl' + nth"></p>
    <p class="p-paperUrl" :ref="'paperUrl' + nth"></p>
    <p class="p-supplementalUrl" :ref="'supplementalUrl' + nth"></p>
    <div class="card">
      <div class="card-header" :id="'accordion' + typedId">
        <h5 class="mb-0">
          <button @click="tracking(content.contentvalue.ID)" class="btn btn-link" data-toggle="collapse"
            :data-target="'#card' + typedId" aria-expanded="true" aria-controls="collapseOne">
            Show More
          </button>
        </h5>
      </div>
      <div :id="'card' + typedId" class="collapse" aria-labelledby="headingOne" :data-parent="'#accordion' + typedId">
        <div class="card-body">
          <p class="p-abstract" :ref="'abstract' + nth"><span style="font-weight:bold"><i
                class="fa-solid fa-quote-left"></i> Abstract </span>: {{ content.contentvalue.Abstract }}</p>
          <p class="p-keywords" :ref="'keywords' + nth"><span style="font-weight:bold"><i class="fa-solid fa-key"></i>
              Keywords </span>: {{ content.contentvalue.Keywords }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import TypeIt from "typeit";
// const { emitEvent } = useEventBus();
import ModelRequest from '@/restful/ModelRequest';
import UserRequest from '@/restful/UserRequest';

// import useEventBus from '@/composables/useEventBus';

export default {
  name: "TypedText",
  props: {
    content: Object,
    nth: Number,
  },
  data() {
    return {
      typedId: `typed-text-${Math.random().toString(36).substr(2, 9)}`
    };
  },
  mounted() {
    console.log(this.content);
    if (this.content.type === 'result') {
      this.showResultDetails();
    }
  },
  methods: {
    showResultDetails() {
      new TypeIt(this.$refs['title' + this.nth], { speed: 1, lifelike: true, cursor: false })
        .type(this.content.contentvalue.Title)
        .exec(() => new TypeIt(this.$refs['authors' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Authors </span>: ${this.content.contentvalue.Authors}`).go())
        .exec(() => new TypeIt(this.$refs['year' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Year </span>: ${this.content.contentvalue.Year}`).go())
        .exec(() => new TypeIt(this.$refs['status' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Status </span>: ${this.content.contentvalue.Status}`).go())
        .exec(() => new TypeIt(this.$refs['bookTitle' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Book Title </span>:  ${this.content.contentvalue['Book Title']}`).go())
        .exec(() => new TypeIt(this.$refs['editors' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Editors </span>: ${this.content.contentvalue.Editors}`).go())
        .exec(() => new TypeIt(this.$refs['publishers' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Publishers </span>: ${this.content.contentvalue.Publishers}`).go())
        .exec(() => new TypeIt(this.$refs['mainUrl' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Main Url </span>: <a target="_blank" style='color:#0069D9' href="${this.content.contentvalue['Main Url']}">${this.content.contentvalue['Main Url']}</a>`).go())
        .exec(() => new TypeIt(this.$refs['paperUrl' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Paper Url </span>: <a target="_blank" style='color:#0069D9' href="${this.content.contentvalue['Paper Url']}">${this.content.contentvalue['Paper Url']}</a>`).go())
        .exec(() => new TypeIt(this.$refs['supplementalUrl' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Supplemental Url </span>: <a target="_blank" style='color:#0069D9' href="${this.content.contentvalue['Supplemental Url']}">${this.content.contentvalue['Supplemental Url']}</a>`).go())
        // .exec(() => new TypeIt(this.$refs['abstract' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Abstract </span>: ${this.content.contentvalue.Abstract}`).go())
        // .exec(() => new TypeIt(this.$refs['keywords' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Keywords </span>: ${this.content.contentvalue.Keywords}`).go())
        .go();
    },
    handleCaseNotResult(inputText) {
      var resultText = inputText
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
      return resultText;
    },
    async tracking(id_paper) {
      var arr_keys = [];
      var totalStr = this.content.contentvalue.Abstract + this.content.contentvalue.Title
      totalStr = totalStr.toLowerCase();
      if (this.content.contentvalue.keywords.length > 0) { // phải là search by model 
        arr_keys = Array.from(this.content.contentvalue.keywords);
        let filteredKeys = arr_keys.filter(item =>
          (totalStr.includes(item)) // chỉ giữ lại những keyword có trong bài báo được click
        );
        var submitData = {
          id_paper: id_paper,
          keywords: filteredKeys
        }
      }
      if (this.content.contentvalue.search_by == 'search_by_keywords') { // hoặc search by keyword 
        arr_keys.push(this.content.contentvalue.question);
        submitData = {
          id_paper: id_paper,
          keywords: arr_keys
        }
      }
      console.log(submitData);
      try {
        const { data, messages } = await ModelRequest.post('tracking', submitData, false);
        console.log('Update data tracking success !');
        console.log(data, messages, 'success');
      } catch (error) {
        console.error('Update data tracking false !', error);
      }
      try {
        const { data, messages } = await UserRequest.post('tracking/add', submitData, false);
        console.log(data, messages, 'laravel success');
      } catch (error) {
        console.error('Update data tracking false !', error);
      }
    }
  }
};
</script>


<style scoped>
.search-by {
  width: 100%;
  display: flex;
  justify-content: start;
  position: relative;
  display: flex;
  justify-content: center;
}

.search-line {
  background-color: #007BFF;
  height: 2px;
  width: 60%;
  position: absolute;
  top: 12px;
}

.search-by-content {
  color: #007BFF;
  z-index: 1;
  text-align: center;
  width: fit-content;
  /* border: 1px solid #ddd; */
  background-color: white;
  padding: 0px 10px;
  margin-bottom: 10px;
  font-weight: bold;
  border-radius: 10px;
  width: 16%;
}


.card-header {
  padding: 0 !important;
}

.p-title {
  color: #28A745;
  font-weight: bold;
  font-size: 20px;
}

.container-question {
  width: 100%;
  display: flex;
  justify-content: end;
}

.additional-content {
  color: red;
  text-align: end;
  width: fit-content;
  border: 1px solid #ddd;
  background-color: #f9f9f9;
  padding: 5px;
  margin-bottom: 10px;
  font-weight: bold;
  border-radius: 10px;
  max-width: 50%;
}

.paper-details {
  margin-bottom: 20px;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 5px;
  background-color: #f9f9f9;
}

.paper-details h2 {
  margin-top: 0;
}

.paper-details p {
  margin: 5px 0;
}

.paper-details a {
  color: #0069D9 !important;
}

.name_keyword_span {
  background-color: #28A644;
  border-radius: 10px;
  padding: 0px 6px;
  margin: 3px;
  color: white;
  font-weight: bold;
}

.modal-dialog {
    max-width: 500px;
}

.container-keywords {
  display: flex;
  flex-wrap: wrap;
}
</style>
