import React, { Component } from "react";
import { Link } from "react-router-dom";
import "./Home.css";

class Home extends Component {
  constructor(props) {
    super(props);
    this.state = {
      file: "",
      previewURL: "",
      result: null,
    };
  }

  // to wake up api server
  componentDidMount() {
    fetch("https://zipnet-api.herokuapp.com/", {
      method: "GET",
    })
      .then((response) => response.json())
      .then((data) => console.log(data));
  }

  handleFileOnChange = (event) => {
    event.preventDefault();
    let reader = new FileReader();
    let file = event.target.files[0];
    reader.onloadend = () => {
      this.setState({
        file: file,
        previewURL: reader.result,
      });
    };
    reader.readAsDataURL(file);
  };

  render() {
    let profile_preview = null;
    if (this.state.file !== "") {
      profile_preview = (
        <img className="preview" src={this.state.previewURL} alt=""></img>
      );
    }
    return (
      <div className="container p-3">
        <div className="py-3">
          <span>
            <i className="fa fa-file-image-o"></i>
          </span>
          &nbsp;
          <span>Import Image</span>
        </div>
        <div className="w-100 input-container my-3">
          <div className="w-100 h-100 d-flex flex-column justify-content-center align-items-center bg-transparent p-3">
            <label className="bg-transparent" htmlFor="file">
              <i
                className="fa-2x fa fa-upload bg-transparent text-dark"
                aria-hidden="true"
              ></i>
            </label>
            <span className="mt-3 bg-transparent text-secondary small">
              Upload a Image File
            </span>
            <input
              type="file"
              id="file"
              accept="image/jpg, image/jpeg, image/png, image/gif"
              name="file"
              onChange={this.handleFileOnChange}
              style={{ display: "none" }}
            ></input>
          </div>
        </div>

        <div className="py-3">
          <span>
            <i className="fa fa-eye"></i>
          </span>
          &nbsp;
          <span>Preview</span>
        </div>
        <div className="w-100 preview-container my-3">
          <div className="w-100 h-100 d-flex flex-column justify-content-center align-items-center bg-transparent px-5">
            {profile_preview}
          </div>
        </div>
        <Link
          to={{
            pathname: "/result",
            state: { file: this.state.file, previewURL: this.state.previewURL },
          }}
          className="w-100 h-100 button-container p-3 d-flex justify-content-center align-items-center bg-white text-decoration-none text-dark"
        >
          <i
            className="fa fa-search bg-transparent text-dark"
            aria-hidden="true"
          ></i>
          &nbsp; Predict
        </Link>
      </div>
    );
  }
}

export default Home;
