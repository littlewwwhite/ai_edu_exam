from pyvis.network import Network
import gradio as gr
import os
from bs4 import BeautifulSoup

# 定义知识图谱的数据
nodes = [
    {"id": "Person A", "label": "Person A", "title": "Person A: Data Scientist\nAge: 30\nLocation: New York"},
    {"id": "Person B", "label": "Person B", "title": "Person B: Researcher\nAge: 32\nLocation: San Francisco"},
    {"id": "Person C", "label": "Person C", "title": "Person C: Engineer\nAge: 28\nLocation: Boston"}
]

edges = [
    {"source": "Person A", "target": "Person B", "title": "Colleagues\nSince: 2020", "weight": 5},
    {"source": "Person B", "target": "Person C", "title": "Friends\nSince: 2015", "weight": 3},
    {"source": "Person A", "target": "Person C", "title": "Family\nSince: Childhood", "weight": 8}
]


# 创建Pyvis网络图对象并生成HTML字符串
def create_knowledge_graph():
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

    # 添加节点和边到网络图中
    for node in nodes:
        net.add_node(node["id"], label=node["label"], title=node["title"], color="#03DAC6", shape="dot", size=15)
    for edge in edges:
        net.add_edge(edge["source"], edge["target"], title=edge["title"], value=edge["weight"], color="#BB86FC",
                     width=edge["weight"])

    # 配置网络图选项
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "shadow": true,
        "color": {
          "highlight": {
            "border": "#03DAC6",
            "background": "#03DAC6"
          }
        }
      },
      "edges": {
        "color": {
          "color": "#BB86FC",
          "highlight": "#FF0266",
          "inherit": false,
          "opacity": 0.9
        },
        "smooth": {
          "type": "dynamic"
        },
        "shadow": true,
        "width": 2,
        "hoverWidth": 3,
        "selectionWidth": 4
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 200,
        "zoomView": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 250,
          "springConstant": 0.001
        }
      }
    }
    """)

    # 生成HTML字符串并且内联所有资源
    html_content = net.generate_html(notebook=False)

    # 将CSS和JS内联到HTML中
    def inline_resources(html):
        soup = BeautifulSoup(html, 'html.parser')

        # 内联 vis-network CSS
        for link in soup.findAll('link', {'rel': 'stylesheet'}):
            css_url = link['href']
            if "vis-network" in css_url:
                with open('static/vis-network.min.css', 'r') as f:
                    style = f.read()
                new_style = soup.new_tag('style')
                new_style.string = style
                link.replace_with(new_style)

        # 内联 vis-network JS
        for script in soup.findAll('script', {'src': True}):
            js_url = script['src']
            if "vis-network" in js_url:
                with open('static/vis-network.min.js', 'r') as f:
                    js_code = f.read()
                script.string = js_code
                del script['src']

        # 添加 Bootstrap CSS
        bootstrap_css_link = soup.new_tag('link', rel='stylesheet', href='static/bootstrap.min.css')
        soup.head.append(bootstrap_css_link)

        # 添加 Bootstrap JS
        bootstrap_js_script = soup.new_tag('script', src='static/bootstrap.bundle.min.js')
        soup.body.append(bootstrap_js_script)

        return str(soup)
    with open('temp_knowledge_graph.html', 'r') as f:
        html_content=f.read()
    # print(html_content)
    return html_content



# 在Gradio中加载图表并显示
def visualize_graph():
    return create_knowledge_graph()


# 使用Gradio创建接口
gr.Interface(
    fn=visualize_graph,
    inputs=[],
    outputs=gr.HTML(label="Knowledge Graph")
).launch(
    server_name="172.18.232.176",
    server_port=7863,
    share=True,
    inbrowser=False,
)