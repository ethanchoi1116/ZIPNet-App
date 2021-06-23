import React, { Component } from "react";
import "./Result.css";

class Home extends Component {
  constructor(props) {
    super(props);
    this.state = {
      p: 0,
      pred: 0,
    };
  }

  componentDidMount() {
    let image = new FormData();
    image.append("file", this.props.location.state.file);
    fetch("https://zipnet-api.herokuapp.com/predict", {
      method: "POST",
      body: image,
    })
      .then((response) => response.json())
      .then((data) => this.setState({ p: data["p"], pred: data["pred"] }));
  }

  render() {
    let profile_preview = (
      <img
        className="preview"
        src={this.props.location.state.previewURL}
        alt=""
      ></img>
    );
    var p = this.state.p;
    var pred = this.state.pred;
    return (
      <div className="container p-3">
        <div className="py-3">
          <span>
            <i className="fa fa-bar-chart"></i>
          </span>
          &nbsp;
          <span>Prediction Result</span>
        </div>
        <div className="w-100 preview-container my-3">
          <div className="w-100 h-100 d-flex flex-column justify-content-center align-items-center bg-transparent px-5">
            {profile_preview}
          </div>
        </div>
        <div className="w-100 h-100 button-container p-3 d-flex justify-content-evenly bg-white text-decoration-none text-dark">
          <div className="bg-transparent small p-3">
            <i className="fa fa-pie-chart bg-transparent "></i>
            &nbsp;
            <i className="fa fa-user-times bg-transparent "></i>
            &nbsp;
            {": " + p.toFixed(2)}
          </div>
          <div className="bg-transparent small p-3">
            <i className="fa fa-users bg-transparent "></i>
            &nbsp;
            {": " + pred.toFixed(2)}
          </div>
        </div>
      </div>
    );
  }
}

export default Home;
