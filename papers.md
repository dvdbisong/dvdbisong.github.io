---
layout: default
title: Paper Reviews
permalink: /papers/
---

<div class="home">
  
  <ul class="posts">
    {% for post in site.categories.[page.title] %}
      <li>
        <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
        <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
        <br>
        {{ post.excerpt }}
      </li>
    {% endfor %}
  </ul>

</div>