<template>
  <div class="container-question" v-if="this.content.type != 'result'">
    <p ref="additionalContent" class="additional-content"></p>
  </div>
  <div v-if="this.content.type == 'result'" :id="typedId" class="paper-details">
    <h2 ref="title"></h2>
    <p ref="authors"></p>
    <p ref="year"></p>
    <p ref="volume"></p>
    <p ref="pages"></p>
    <p ref="status"></p>
    <p ref="bookTitle"></p>
    <p ref="editors"></p>
    <p ref="publishers"></p>
    <p ref="mainUrl"></p>
    <p ref="paperUrl"></p>
    <p ref="supplementalUrl"></p>
    <p ref="abstract"></p>
    <p ref="keywords"></p>
  </div>
</template>

<script>
import TypeIt from "typeit";

export default {
  name: "TypedText",
  props: {
    content: Object,
  },
  data() {
    return {
      typedId: `typed-text-${Math.random().toString(36).substr(2, 9)}`
    };
  },
  mounted() {
    // Kiểm tra nếu content là kiểu result thì hiển thị thông tin của kết quả
    if (this.content.type === 'result') {
      this.showResultDetails();
    } else {
      this.showAdditionalContent();
      // Nếu content không phải là kiểu result, có thể là câu hỏi hoặc loại khác, bạn có thể xử lý tùy ý
    }
  },
  methods: {
    showAdditionalContent() {
      if (this.$refs.additionalContent) {
        this.$refs.additionalContent.innerHTML  = '<i class="fa-solid fa-circle-question"></i> ' + this.content.contentvalue;
      }
    },
    showResultDetails() {
      new TypeIt(this.$refs.title, { speed: 1, lifelike: false, cursor: false })
        .type(this.content.contentvalue.Title)
        .exec(() => new TypeIt(this.$refs.authors, { speed: 1, lifelike: false, cursor: false }).type(`Authors: ${this.content.contentvalue.Authors}`).go())
        .exec(() => new TypeIt(this.$refs.year, { speed: 1, lifelike: false, cursor: false }).type(`Year: ${this.content.contentvalue.Year}`).go())
        // .exec(() => new TypeIt(this.$refs.volume, { speed: 1, lifelike: false, cursor: false }).type(`Volume: ${this.content.contentvalue.Volume}`).go())
        // .exec(() => new TypeIt(this.$refs.pages, { speed: 1, lifelike: false, cursor: false }).type(`Pages: ${this.content.contentvalue.Pages}`).go())
        .exec(() => new TypeIt(this.$refs.status, { speed: 1, lifelike: false, cursor: false }).type(`Status: ${this.content.contentvalue.Status}`).go())
        .exec(() => new TypeIt(this.$refs.bookTitle, { speed: 1, lifelike: false, cursor: false }).type(`Book Title: ${this.content.contentvalue['Book Title']}`).go())
        .exec(() => new TypeIt(this.$refs.editors, { speed: 1, lifelike: false, cursor: false }).type(`Editors: ${this.content.contentvalue.Editors}`).go())
        .exec(() => new TypeIt(this.$refs.publishers, { speed: 1, lifelike: false, cursor: false }).type(`Publishers: ${this.content.contentvalue.Publishers}`).go())
        .exec(() => new TypeIt(this.$refs.mainUrl, { speed: 1, lifelike: false, cursor: false }).type(`Main Url: <a style='color:#0069D9' href="${this.content.contentvalue['Main Url']}">${this.content.contentvalue['Main Url']}</a>`).go())
        .exec(() => new TypeIt(this.$refs.paperUrl, { speed: 1, lifelike: false, cursor: false }).type(`Paper Url: <a style='color:#0069D9' href="${this.content.contentvalue['Paper Url']}">${this.content.contentvalue['Paper Url']}</a>`).go())
        .exec(() => new TypeIt(this.$refs.supplementalUrl, { speed: 1, lifelike: false, cursor: false }).type(`Supplemental Url: <a style='color:#0069D9' href="${this.content.contentvalue['Supplemental Url']}">${this.content.contentvalue['Supplemental Url']}</a>`).go())
        // .exec(() => new TypeIt(this.$refs.abstract, { speed: 1, lifelike: false, cursor: false }).type(`Abstract: ${this.content.contentvalue.Abstract}`).go())
        // .exec(() => new TypeIt(this.$refs.keywords, { speed: 1, lifelike: false, cursor: false }).type(`Keywords: ${this.content.contentvalue.Keywords}`).go())
        .go();
    }
  }
};
</script>

<style scoped>
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
