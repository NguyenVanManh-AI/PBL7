<template>
  <div class="container-question" v-if="content.type != 'result'">
    <p :ref="'additionalContent' + nth" class="additional-content"></p>
  </div>
  <div v-if="content.type == 'result'" :id="typedId" class="paper-details">
    <p class="p-title"><span style='font-weight:bold'><i class="fa-solid fa-bookmark"></i></span> <span :ref="'title' + nth"></span></p>
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
    <p class="p-abstract" :ref="'abstract' + nth"></p>
    <p class="p-keywords" :ref="'keywords' + nth"></p>
  </div>
</template>

<script>
import TypeIt from "typeit";

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
    if (this.content.type === 'result') {
      this.showResultDetails();
    } else {
      this.showAdditionalContent();
    }
  },
  methods: {
    showAdditionalContent() {
      if (this.$refs['additionalContent' + this.nth]) {
        this.$refs['additionalContent' + this.nth].innerHTML = '<i class="fa-solid fa-circle-question"></i> ' + this.content.contentvalue;
      }
    },
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
        .exec(() => new TypeIt(this.$refs['abstract' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Abstract </span>: ${this.content.contentvalue.Abstract}`).go())
        .exec(() => new TypeIt(this.$refs['keywords' + this.nth], { speed: 1, lifelike: true, cursor: false }).type(`<span style='font-weight:bold'>Keywords </span>: ${this.content.contentvalue.Keywords}`).go())
        .go();
    }
  }
};
</script>


<style scoped>
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
</style>
