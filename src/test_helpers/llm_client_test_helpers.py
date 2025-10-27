# test_variables.py

from typing import Literal
import json
import random
import uuid

from langchain_core.messages import AIMessage

expected_test_responses = {
    "isolate_vehicle_description": {
        "success": {
            "description": (
                "Take on any road with confidence in this 2010 Subaru Outback 3.6R AWD, "
                "a versatile crossover SUV known for its legendary all-wheel-drive capability "
                "and dependability. Perfect for daily driving, weekend adventures, or family trips, "
                "the Outback offers comfort, utility, and Subaru reliability."
            )
        },
        "failed": {"description": "No description found."},
        "unexpected_json": {"vehicle_info": "It's a great car!"},
        "not_json": "It's an OK car!"
    }
}

def create_mock_llm_response(
    function_name: Literal["isolate_vehicle_description"],
    provider: Literal["anthropic"],
    response_type: Literal["success", "failed", "unexpected_json", "not_json"] = "success"
) -> AIMessage:
    """
    Create a simulated AIMessage to mimic LLM responses with realistic structure per provider.
    """
    try:
        content_value = expected_test_responses[function_name][response_type]
    except KeyError:
        content_value = "Generic response"

    # Convert dict responses to JSON string; leave strings as-is
    content = json.dumps(content_value) if isinstance(content_value, dict) else content_value

    # --- token counts ---
    input_tokens = random.randint(50, 150)
    output_tokens = random.randint(20, 100)
    total_tokens = input_tokens + output_tokens

    # --- build response metadata depending on provider ---
    if provider == "anthropic":
        response_metadata = {
            "id": str(uuid.uuid4()),
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        }
    else:
        raise ValueError(f"Unknown llm provider: {provider}")

    return AIMessage(
        content=content,
        additional_kwargs={},
        response_metadata=response_metadata,
        id=str(uuid.uuid4()),
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
    )



TEST_VDP_URLS = [
    {
        # BUICK_VERAN
        "url": "https://www.greghublerford.com/inventory/used-2016-buick-verano-sport-touring-group-fwd-4d-sedan-1g4pw5sk8g4135010/",
        "description": (
            """
            Greg Hubler Ford of Muncie is proud to offer this 2016 Buick Verano in Graphite Gray Metallic with Medium Titanium Artificial Leather.

            Well-equipped and stylish, this 2016 Buick Verano Sport Touring combines comfort, safety, and performance in one sleek package. Finished in Graphite Gray Metallic with premium synthetic seating, it features a 2.4L I4 engine, smooth 6-speed automatic transmission, and front-wheel drive for dependable performance.

            Key Features & Options:

            Sport Touring Edition with rear spoiler, aluminum wheels, and performance tires.
            Sunroof/Moonroof.
            Premium synthetic leather-trimmed seats & leather-wrapped steering wheel.
            Heated front seats & power drivers seat.
            Remote engine start & keyless entry.
            Multi-zone climate control with air conditioning.
            Satellite radio, Bluetooth, MP3/Aux input.

            Safety & Driver Assistance:
            Blind Spot Monitor & Lane Departure Warning
            Forward Collision Alert & Cross-Traffic Alert
            Rearview camera with parking sensors
            Full airbag system (front, side, rear, and knee protection)
            Stability & traction control, ABS brakes, tire pressure monitoring

            Technology:
            Built-in Navigation
            Telematics & Wi-Fi hotspot capability
            Universal garage door opener

            This Verano has desirable upgrades like the Driver Confidence Package and Sun/Moonroof. Clean inside and out, it offers a refined ride with advanced safety and tech features.

            Introducing the All New Greg Hubler Promise: The Promise. That's what you get when you purchase a new or pre-owned vehicle from Greg Hubler Ford So just what is the Greg Hubler Promise? Our New Vehicle Promise: With every new Ford purchase, you'll receive three years or forty five thousand miles of scheduled maintenance (oil changes and tire rotations). Our Used Vehicle Promise: With most pre-driven vehicle purchases, you'll receive a 12 month, 12 thousand-mile warranty as well as free oil changes and tire rotations for one year for most vehicles under 100,000 miles and less than 10 model years old at the time of sale. In addition, for both new and used purchases, The Promise gives you complimentary pick up and drop off when you bring your vehicle in for service and a free car wash and vacuum whenever you stop in! The Greg Hubler Ford Promise: it's Peace of Mind for you. Promise excludes Supplier customers. See dealer for details. Visit Greg Hubler Ford conveniently located at 6400 W Hometown Blvd. Muncie, Indiana 47304. We stock a wide selection of new and pre-owned cars / trucks / vans / SUVs with the most competitive pricing in the area. Visit our website at www.GregHublerFord.com or call us at 765 289-0431.
            """
        )
    },
    # FORD FUSION
    {
        "url": "https://www.greghublerford.com/inventory/used-2016-ford-fusion-se-fwd-4d-sedan-3fa6p0h77gr192119/",
        "description": (
            """
            Greg Hubler Ford of Muncie is proud to offer this 2016 Ford Fusion in with Cloth.

            Sporty, fuel-efficient, and loaded with features, this 2016 Ford Fusion SE is a well-rounded midsize sedan. Powered by a 1.5L EcoBoost I4 engine paired with a 6-speed automatic transmission, it delivers both performance and efficiency with up to 34 MPG highway.

            Key Features & Options:

            Ruby Red Tinted Clearcoat exterior with Charcoal Black cloth interior

            SE Appearance Package with 18" 5-spoke black premium wheels & rear spoiler
            Reverse sensing system & rearview camera
            Heated front seats & remote start (SE Cold Weather Package)
            Dual-zone automatic climate control
            SYNC with MyFord, SiriusXM capability, Bluetooth & audio controls on steering wheel

            Comfort & Convenience:
            Power drivers seat
            60/40 split-fold rear seat
            Keyless entry & push-button start
            Power windows, locks, mirrors
            All-weather front & rear floor mats

            Safety & Security:
            4-wheel disc brakes with ABS
            Advanced airbag system, driver knee airbag
            Traction control, stability control
            Tire pressure monitoring system
            Perimeter alarm & SecuriCode keyless keypad
            Fuel Economy:

            26 MPG combined (22 city / 34 highway)

            A Top Safety Pick, its a reliable daily driver with modern tech and comfort.

            Introducing the All New Greg Hubler Promise: The Promise. That's what you get when you purchase a new or pre-owned vehicle from Greg Hubler Ford So just what is the Greg Hubler Promise? Our New Vehicle Promise: With every new Ford purchase, you'll receive three years or forty five thousand miles of scheduled maintenance (oil changes and tire rotations). Our Used Vehicle Promise: With most pre-driven vehicle purchases, you'll receive a 12 month, 12 thousand-mile warranty as well as free oil changes and tire rotations for one year for most vehicles under 100,000 miles and less than 10 model years old at the time of sale. In addition, for both new and used purchases, The Promise gives you complimentary pick up and drop off when you bring your vehicle in for service and a free car wash and vacuum whenever you stop in! The Greg Hubler Ford Promise: it's Peace of Mind for you. Promise excludes Supplier customers. See dealer for details. Visit Greg Hubler Ford conveniently located at 6400 W Hometown Blvd. Muncie, Indiana 47304. We stock a wide selection of new and pre-owned cars / trucks / vans / SUVs with the most competitive pricing in the area. Visit our website at www.GregHublerFord.com or call us at 765 289-0431.
            """
        )
    }
]